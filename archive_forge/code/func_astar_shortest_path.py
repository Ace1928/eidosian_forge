import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def astar_shortest_path(graph, node, goal_fn, edge_cost_fn, estimate_cost_fn):
    """Compute the A* shortest path for a graph

    :param graph: The input graph to use. Can
        either be a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`
    :param int node: The node index to compute the path from
    :param goal_fn: A python callable that will take in 1 parameter, a node's
        data object and will return a boolean which will be True if it is the
        finish node.
    :param edge_cost_fn: A python callable that will take in 1 parameter, an
        edge's data object and will return a float that represents the cost
        of that edge. It must be non-negative.
    :param estimate_cost_fn: A python callable that will take in 1 parameter, a
        node's data object and will return a float which represents the
        estimated cost for the next node. The return must be non-negative. For
        the algorithm to find the actual shortest path, it should be
        admissible, meaning that it should never overestimate the actual cost
        to get to the nearest goal node.

    :returns: The computed shortest path between node and finish as a list
        of node indices.
    :rtype: NodeIndices
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))