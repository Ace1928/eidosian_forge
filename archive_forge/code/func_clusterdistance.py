import numbers
from . import _cluster  # type: ignore
def clusterdistance(self, index1=0, index2=0, method='a', dist='e', transpose=False):
    """Calculate the distance between two clusters.

        Keyword arguments:
         - index1: 1D array identifying which genes/samples belong to the
           first cluster. If the cluster contains only one gene, then
           index1 can also be written as a single integer.
         - index2: 1D array identifying which genes/samples belong to the
           second cluster. If the cluster contains only one gene, then
           index2 can also be written as a single integer.
         - transpose: if False, genes (rows) are clustered;
                      if True, samples (columns) are clustered.
         - dist: specifies the distance function to be used:
           - dist == 'e': Euclidean distance
           - dist == 'b': City Block distance
           - dist == 'c': Pearson correlation
           - dist == 'a': absolute value of the correlation
           - dist == 'u': uncentered correlation
           - dist == 'x': absolute uncentered correlation
           - dist == 's': Spearman's rank correlation
           - dist == 'k': Kendall's tau
         - method: specifies how the distance between two clusters is defined:
           - method == 'a': the distance between the arithmetic means
           of the two clusters
           - method == 'm': the distance between the medians of the
           two clusters
           - method == 's': the smallest pairwise distance between members
           of the two clusters
           - method == 'x': the largest pairwise distance between members
           of the two clusters
           - method == 'v': average of the pairwise distances between members
           of the two clusters
         - transpose: if False: clusters of rows are considered;
                      if True: clusters of columns are considered.
        """
    if transpose:
        weight = self.gweight
    else:
        weight = self.eweight
    return clusterdistance(self.data, self.mask, weight, index1, index2, method, dist, transpose)