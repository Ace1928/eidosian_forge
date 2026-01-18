import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def is_subgraph_isomorphic(first, second, node_matcher=None, edge_matcher=None, id_order=False, induced=True, call_limit=None):
    """Determine if 2 graphs are subgraph isomorphic

    This checks if 2 graphs are subgraph isomorphic both structurally and also
    comparing the node and edge data using the provided matcher functions.
    The matcher functions take in 2 data objects and will compare them.
    Since there is an ambiguity in the term 'subgraph', do note that we check
    for an node-induced subgraph if argument `induced` is set to `True`. If it is
    set to `False`, we check for a non induced subgraph, meaning the second graph
    can have fewer edges than the subgraph of the first. By default it's `True`. A
    simple example that checks if they're just equal would be::

            graph_a = rustworkx.PyGraph()
            graph_b = rustworkx.PyGraph()
            rustworkx.is_subgraph_isomorphic(graph_a, graph_b,
                                            lambda x, y: x == y)


    :param first: The first graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable node_matcher: A python callable object that takes 2
        positional one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are viewed
        as matching.
    :param callable edge_matcher: A python callable object that takes 2
        positional one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``True`` this function will match the nodes
        in order specified by their ids. Otherwise it will default to a heuristic
        matching order based on [VF2]_ paper.
    :param bool induced: If set to ``True`` this function will check the existence
        of a node-induced subgraph of first isomorphic to second graph.
        Default: ``True``.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop and return ``False``.

    :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`
        , ``False`` if there is not.
    :rtype: bool
    """
    raise TypeError('Invalid Input Type %s for graph' % type(first))