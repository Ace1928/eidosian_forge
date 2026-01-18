import numbers
from . import _cluster  # type: ignore
def distancematrix(self, transpose=False, dist='e'):
    """Calculate the distance matrix and return it as a list of arrays.

        Keyword arguments:
         - transpose:
             if False: calculate the distances between genes (rows);
             if True: calculate the distances between samples (columns).
         - dist: specifies the distance function to be used:
           - dist == 'e': Euclidean distance
           - dist == 'b': City Block distance
           - dist == 'c': Pearson correlation
           - dist == 'a': absolute value of the correlation
           - dist == 'u': uncentered correlation
           - dist == 'x': absolute uncentered correlation
           - dist == 's': Spearman's rank correlation
           - dist == 'k': Kendall's tau

        Return value:

        The distance matrix is returned as a list of 1D arrays containing the
        distance matrix between the gene expression data. The number of columns
        in each row is equal to the row number. Hence, the first row has zero
        length. An example of the return value is:

            matrix = [[],
                      array([1.]),
                      array([7., 3.]),
                      array([4., 2., 6.])]

        This corresponds to the distance matrix:

            [0., 1., 7., 4.]
            [1., 0., 3., 2.]
            [7., 3., 0., 6.]
            [4., 2., 6., 0.]

        """
    if transpose:
        weight = self.gweight
    else:
        weight = self.eweight
    return distancematrix(self.data, self.mask, weight, transpose, dist)