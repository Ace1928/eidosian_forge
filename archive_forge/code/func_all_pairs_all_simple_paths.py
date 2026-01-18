import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def all_pairs_all_simple_paths(graph, min_depth=None, cutoff=None):
    """Return all the simple paths between all pairs of nodes in the graph

    This function is multithreaded and will launch a thread pool with threads
    equal to the number of CPUs by default. You can tune the number of threads
    with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
    ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.

    :param graph: The graph to find all simple paths in. This can be a :class:`~rustworkx.PyGraph`
        or a :class:`~rustworkx.PyDiGraph`
    :param int min_depth: The minimum depth of the path to include in the output
        list of paths. By default all paths are included regardless of depth,
        setting to 0 will behave like the default.
    :param int cutoff: The maximum depth of path to include in the output list
        of paths. By default includes all paths regardless of depth, setting to
        0 will behave like default.

    :returns: A mapping of source node indices to a mapping of target node
        indices to a list of paths between the source and target nodes.
    :rtype: AllPairsMultiplePathMapping
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))