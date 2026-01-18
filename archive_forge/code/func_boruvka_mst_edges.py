from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight', preserve_edge_attrs='data')
def boruvka_mst_edges(G, minimum=True, weight='weight', keys=False, data=True, ignore_nan=False):
    """Iterate over edges of a Borůvka's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The edges of `G` must have distinct weights,
        otherwise the edges may not form a tree.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        This argument is ignored since this function is not
        implemented for multigraphs; it exists only for consistency
        with the other minimum spanning tree functions.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    forest = UnionFind(G)

    def best_edge(component):
        """Returns the optimum (minimum or maximum) edge on the edge
        boundary of the given set of nodes.

        A return value of ``None`` indicates an empty boundary.

        """
        sign = 1 if minimum else -1
        minwt = float('inf')
        boundary = None
        for e in nx.edge_boundary(G, component, data=True):
            wt = e[-1].get(weight, 1) * sign
            if isnan(wt):
                if ignore_nan:
                    continue
                msg = f'NaN found as an edge weight. Edge {e}'
                raise ValueError(msg)
            if wt < minwt:
                minwt = wt
                boundary = e
        return boundary
    best_edges = (best_edge(component) for component in forest.to_sets())
    best_edges = [edge for edge in best_edges if edge is not None]
    while best_edges:
        best_edges = (best_edge(component) for component in forest.to_sets())
        best_edges = [edge for edge in best_edges if edge is not None]
        for u, v, d in best_edges:
            if forest[u] != forest[v]:
                if data:
                    yield (u, v, d)
                else:
                    yield (u, v)
                forest.union(u, v)