import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def longest_simple_path(graph):
    """Return a longest simple path in the graph

    This function searches computes all pairs of all simple paths and returns
    a path of the longest length from that set. It is roughly equivalent to
    running something like::

        from rustworkx import all_pairs_all_simple_paths

        max((y.values for y in all_pairs_all_simple_paths(graph).values()), key=lambda x: len(x))

    but this function will be more efficient than using ``max()`` as the search
    is evaluated in parallel before returning to Python. In the case of multiple
    paths of the same maximum length being present in the graph only one will be
    provided. There are no guarantees on which of the multiple longest paths
    will be returned (as it is determined by the parallel execution order). This
    is a tradeoff to improve runtime performance. If a stable return is required
    in such case consider using the ``max()`` equivalent above instead.

    This function is multithreaded and will launch a thread pool with threads
    equal to the number of CPUs by default. You can tune the number of threads
    with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
    ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.

    :param PyGraph graph: The graph to find the longest path in

    :returns: A sequence of node indices that represent the longest simple graph
        found in the graph. If the graph is empty ``None`` will be returned instead.
    :rtype: NodeIndices
    """