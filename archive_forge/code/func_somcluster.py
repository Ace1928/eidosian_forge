import numbers
from . import _cluster  # type: ignore
def somcluster(self, transpose=False, nxgrid=2, nygrid=1, inittau=0.02, niter=1, dist='e'):
    """Calculate a self-organizing map on a rectangular grid.

        The somcluster method returns a tuple (clusterid, celldata).

        Keyword arguments:
         - transpose: if False, genes (rows) are clustered;
                      if True,  samples (columns) are clustered.
         - nxgrid: the horizontal dimension of the rectangular SOM map
         - nygrid: the vertical dimension of the rectangular SOM map
         - inittau: the initial value of tau (the neighborbood function)
         - niter: the number of iterations
         - dist: specifies the distance function to be used:
           - dist == 'e': Euclidean distance
           - dist == 'b': City Block distance
           - dist == 'c': Pearson correlation
           - dist == 'a': absolute value of the correlation
           - dist == 'u': uncentered correlation
           - dist == 'x': absolute uncentered correlation
           - dist == 's': Spearman's rank correlation
           - dist == 'k': Kendall's tau

        Return values:
         - clusterid: array with two columns, while the number of rows is equal
           to the number of genes or the number of samples depending on
           whether genes or samples are being clustered. Each row in
           the array contains the x and y coordinates of the cell in the
           rectangular SOM grid to which the gene or samples was assigned.
         - celldata: an array with dimensions (nxgrid, nygrid, number of
           samples) if genes are being clustered, or (nxgrid, nygrid,
           number of genes) if samples are being clustered. Each item
           [ix, iy] of this array is a 1D vector containing the gene
           expression data for the centroid of the cluster in the SOM grid
           cell with coordinates [ix, iy].
        """
    if transpose:
        weight = self.gweight
    else:
        weight = self.eweight
    return somcluster(self.data, self.mask, weight, transpose, nxgrid, nygrid, inittau, niter, dist)