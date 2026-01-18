import one of the named maximum matching algorithms directly.
import collections
import itertools
import networkx as nx
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
@nx._dispatch
def eppstein_matching(G, top_nodes=None):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matching`, such that
      ``matching[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matching`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with David Eppstein's version of the algorithm
    Hopcroft--Karp algorithm (see :func:`hopcroft_karp_matching`), which
    originally appeared in the `Python Algorithms and Data Structures library
    (PADS) <http://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt>`_.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------

    hopcroft_karp_matching

    """
    left, right = bipartite_sets(G, top_nodes)
    G = nx.DiGraph(G.edges(left))
    matching = {}
    for u in G:
        for v in G[u]:
            if v not in matching:
                matching[v] = u
                break
    while True:
        preds = {}
        unmatched = []
        pred = {u: unmatched for u in G}
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)
        while layer and (not unmatched):
            newLayer = {}
            for u in layer:
                for v in G[u]:
                    if v not in preds:
                        newLayer.setdefault(v, []).append(u)
            layer = []
            for v in newLayer:
                preds[v] = newLayer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)
        if not unmatched:
            for key in matching.copy():
                matching[matching[key]] = key
            return matching

        def recurse(v):
            if v in preds:
                L = preds.pop(v)
                for u in L:
                    if u in pred:
                        pu = pred.pop(u)
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False
        for v in unmatched:
            recurse(v)