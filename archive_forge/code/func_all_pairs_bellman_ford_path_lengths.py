import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def all_pairs_bellman_ford_path_lengths(graph, edge_cost_fn):
    """For each node in the graph, calculates the lengths of the shortest paths to all others.

    This function will generate the shortest path lengths from all nodes in the
    graph using the Bellman-Ford algorithm. This function is multithreaded and will
    launch a thread pool with threads equal to the number of CPUs by
    default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    :param graph: The input graph to use. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param edge_cost_fn: A callable object that acts as a weight function for
        an edge. It will accept a single positional argument, the edge's weight
        object and will return a float which will be used to represent the
        weight/cost of the edge

    :return: A read-only dictionary of path lengths. The keys are the source
        node indices and the values are a dict of the target node and the
        length of the shortest path to that node. For example::

            {
                0: {1: 2.0, 2: 2.0},
                1: {2: 1.0},
                2: {0: 1.0},
            }

    :rtype: AllPairsPathLengthMapping

    :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
        path is not defined
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))