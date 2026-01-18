import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
@py_random_state(4)
@nx._dispatch(edge_attrs={'attr': 'default'})
def greedy_branching(G, attr='weight', default=1, kind='max', seed=None):
    """
    Returns a branching obtained through a greedy algorithm.

    This algorithm is wrong, and cannot give a proper optimal branching.
    However, we include it for pedagogical reasons, as it can be helpful to
    see what its outputs are.

    The output is a branching, and possibly, a spanning arborescence. However,
    it is not guaranteed to be optimal in either case.

    Parameters
    ----------
    G : DiGraph
        The directed graph to scan.
    attr : str
        The attribute to use as weights. If None, then each edge will be
        treated equally with a weight of 1.
    default : float
        When `attr` is not None, then if an edge does not have that attribute,
        `default` specifies what value it should take.
    kind : str
        The type of optimum to search for: 'min' or 'max' greedy branching.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    B : directed graph
        The greedily obtained branching.

    """
    if kind not in KINDS:
        raise nx.NetworkXException('Unknown value for `kind`.')
    if kind == 'min':
        reverse = False
    else:
        reverse = True
    if attr is None:
        attr = random_string(seed=seed)
    edges = [(u, v, data.get(attr, default)) for u, v, data in G.edges(data=True)]
    try:
        edges.sort(key=itemgetter(2, 0, 1), reverse=reverse)
    except TypeError:
        edges.sort(key=itemgetter(2), reverse=reverse)
    B = nx.DiGraph()
    B.add_nodes_from(G)
    uf = nx.utils.UnionFind()
    for i, (u, v, w) in enumerate(edges):
        if uf[u] == uf[v]:
            continue
        elif B.in_degree(v) == 1:
            continue
        else:
            data = {}
            if attr is not None:
                data[attr] = w
            B.add_edge(u, v, **data)
            uf.union(u, v)
    return B