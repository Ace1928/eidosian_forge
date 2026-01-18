import numbers
from . import _cluster  # type: ignore
def kmedoids(distance, nclusters=2, npass=1, initialid=None):
    """Perform k-medoids clustering.

    This function performs k-medoids clustering, and returns the cluster
    assignments, the within-cluster sum of distances of the optimal
    k-medoids clustering solution, and the number of times the optimal
    solution was found.

    Keyword arguments:
     - distance: The distance matrix between the items. There are three
       ways in which you can pass a distance matrix:
       1. a 2D NumPy array (in which only the left-lower part of the array
       will be accessed);
       2. a 1D NumPy array containing the distances consecutively;
       3. a list of rows containing the lower-triangular part of
       the distance matrix.

       Examples are:

           >>> from numpy import array
           >>> # option 1:
           >>> distance = array([[0.0, 1.1, 2.3],
           ...                   [1.1, 0.0, 4.5],
           ...                   [2.3, 4.5, 0.0]])
           >>> # option 2:
           >>> distance = array([1.1, 2.3, 4.5])
           >>> # option 3:
           >>> distance = [array([]),
           ...             array([1.1]),
           ...             array([2.3, 4.5])]


       These three correspond to the same distance matrix.
     - nclusters: number of clusters (the 'k' in k-medoids)
     - npass: the number of times the k-medoids clustering algorithm
       is performed, each time with a different (random) initial
       condition.
     - initialid: the initial clustering from which the algorithm should start.
       If initialid is not given, the routine carries out npass
       repetitions of the EM algorithm, each time starting from a
       different random initial clustering. If initialid is given,
       the routine carries out the EM algorithm only once, starting
       from the initial clustering specified by initialid and
       without randomizing the order in which items are assigned to
       clusters (i.e., using the same order as in the data matrix).
       In that case, the k-medoids algorithm is fully deterministic.

    Return values:
     - clusterid: array containing the index of the cluster to which each
       item was assigned in the best k-medoids clustering solution that was
       found in the npass runs; note that the index of a cluster is the index
       of the item that is the medoid of the cluster;
     - error: the within-cluster sum of distances for the returned k-medoids
       clustering solution;
     - nfound: the number of times this solution was found.
    """
    distance = __check_distancematrix(distance)
    nitems = len(distance)
    clusterid, npass = __check_initialid(initialid, npass, nitems)
    error, nfound = _cluster.kmedoids(distance, nclusters, npass, clusterid)
    return (clusterid, error, nfound)