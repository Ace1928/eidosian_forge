import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def floyd_warshall_successor_and_distance(graph, weight_fn=None, default_weight=1.0, parallel_threshold=300):
    """
    Find all-pairs shortest path lengths using Floyd's algorithm.

    Floyd's algorithm is used for finding shortest paths in dense graphs
    or graphs with negative weights (where Dijkstra's algorithm fails).

    This function is multithreaded and will launch a pool with threads equal
    to the number of CPUs by default if the number of nodes in the graph is
    above the value of ``parallel_threshold`` (it defaults to 300).
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads if parallelization was enabled.

    :param PyDiGraph graph: The directed graph to run Floyd's algorithm on
    :param weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells rustworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            floyd_warshall_successor_and_distance(graph, weight_fn=lambda _: 1)

        to return a weight of 1 for all edges. Also:

            floyd_warshall_successor_and_distance(graph, weight_fn=float)

        to cast the edge object as a float as the weight.
    :param as_undirected: If set to true each directed edge will be treated as
        bidirectional/undirected.
    :param int parallel_threshold: The number of nodes to execute
        the algorithm in parallel at. It defaults to 300, but this can
        be tuned

    :returns: A tuple of two matrices.
        First one is a matrix of shortest path distances between nodes. If there is no
        path between two nodes then the corresponding matrix entry will be
        ``np.inf``.
        Second one is a matrix of **next** nodes for given source and target. If there is no
        path between two nodes then the corresponding matrix entry will be the same as
        a target node. To reconstruct the shortest path among nodes::

            def reconstruct_path(source, target, successors):
                path = []
                if source == target:
                    return path
                curr = source
                while curr != target:
                    path.append(curr)
                    curr = successors[curr, target]
                path.append(target)
                return path

    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))