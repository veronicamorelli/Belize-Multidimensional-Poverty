library(dplyr)
library(kohonen)
library(ggplot2)
library(factoextra)

df <- read.csv("BELIZE_hh_nomiss.csv")

df <- df[, -1]

df %>% glimpse()

str(df)

df_som = df[, -c(1, 12, 13)]

str(df_som)

unique_values <- sapply(df_som, unique)
print(unique_values)

######## Unsupervised SOM ######## 

# make a train data sets that scaled and convert them to be a matrix cause 
# kohonen function accept numeric matrix
df_som.train <- as.matrix(scale(df_som[,-1]))

# make stable sampling
RNGkind(sample.kind = "Rounding")
# make a SOM grid
set.seed(100)
df_som.grid <- somgrid(xdim = 10, ydim = 10, topo = "hexagonal")

# make a SOM model
set.seed(100)
df_som.model <- som(df_som.train, df_som.grid, rlen = 500, radius = 2.5, keep.data = TRUE,
                 dist.fcts = "euclidean")
# TO CHANGE METRIC!!
str(df_som.model)

head(df_som.model$unit.classif)

# Mapping Plot
plot(df_som.model, type = "mapping", pchs = 19, shape = "round")

head(data.frame(df_som.train), 5)

# Codes plot
plot(df_som.model, type = "codes", main = "Codes Plot", palette.name = rainbow)

# Training Process
plot(df_som.model, type = "changes")

# Node Counts
plot(df_som.model, type = "counts")

# Neighbours Nodes
plot(df_som.model, type = "dist.neighbours")

# Heatmaps
heatmap.som <- function(model){
  for (i in 1:10) {
    plot(model, type = "property", property = getCodes(model)[,i], 
         main = colnames(getCodes(model))[i]) 
  }
}
heatmap.som(df_som.model)

### Clustering
set.seed(100)
fviz_nbclust(df_som.model$codes[[1]], kmeans, method = "wss")
# 6 clusters

set.seed(100)
clust <- kmeans(df_som.model$codes[[1]], 6)

# clustering using hierarchial
# cluster.som <- cutree(hclust(dist(ads.model$codes[[1]])), 6)

plot(df_som.model, type = "codes", bgcol = rainbow(9)[clust$cluster], main = "Cluster Map")
add.cluster.boundaries(df_som.model, clust$cluster)

# know cluster each data
df_som.cluster <- data.frame(df_som, cluster = clust$cluster[df_som.model$unit.classif])
tail(df_som.cluster, 10)

##### Supervised SOM #####

# split data
set.seed(100)
int <- sample(nrow(df_som), nrow(df_som)*0.8)
train <- df_som[int,]
test <- df_som[-int,]

# scaling data
trainX <- scale(train[,-1])
testX <- scale(test[,-1], center = attr(trainX, "scaled:center"))

# make label
train.label <- factor(train[,1])
test.label <- factor(test[,1])
test[,1] <- 916
testXY <- list(independent = testX, dependent = test.label)

# classification & predict
set.seed(100)
class <- xyf(trainX, classvec2classmat(train.label), df_som.grid, rlen = 500)

plot(class, type = "changes")

# Predict
pred <- predict(class, newdata = testXY)
table(Predict = pred$predictions[[2]], Actual = test.label)

# Cluster boundaries
plot(df_som.model, type = "codes", bgcol = rainbow(9)[clust$cluster], main = "Cluster SOM")
add.cluster.boundaries(df_som.model, clust$cluster)

c.class <- kmeans(class$codes[[2]], 3)
par(mfrow = c(1,2))
plot(class, type = "codes", main = c("Unsupervised SOM", "Supervised SOM"), 
     bgcol = rainbow(3)[c.class$cluster])
add.cluster.boundaries(class, c.class$cluster)
