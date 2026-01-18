import networkx as nx
from ...utils import not_implemented_for
from ..matching import maximal_matching
@not_implemented_for('directed')
@nx._dispatch(node_attrs='weight')
def min_weighted_dominating_set(G, weight=None):
    """Returns a dominating set that approximates the minimum weight node
    dominating set.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph.

    weight : string
        The node attribute storing the weight of an node. If provided,
        the node attribute with this key must be a number for each
        node. If not provided, each node is assumed to have weight one.

    Returns
    -------
    min_weight_dominating_set : set
        A set of nodes, the sum of whose weights is no more than `(\\log
        w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of
        each node in the graph and `w(V^*)` denotes the sum of the
        weights of each node in the minimum weight dominating set.

    Notes
    -----
    This algorithm computes an approximate minimum weighted dominating
    set for the graph `G`. The returned solution has weight `(\\log
    w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of each
    node in the graph and `w(V^*)` denotes the sum of the weights of
    each node in the minimum weight dominating set for the graph.

    This implementation of the algorithm runs in $O(m)$ time, where $m$
    is the number of edges in the graph.

    References
    ----------
    .. [1] Vazirani, Vijay V.
           *Approximation Algorithms*.
           Springer Science & Business Media, 2001.

    """
    if len(G) == 0:
        return set()
    dom_set = set()

    def _cost(node_and_neighborhood):
        """Returns the cost-effectiveness of greedily choosing the given
        node.

        `node_and_neighborhood` is a two-tuple comprising a node and its
        closed neighborhood.

        """
        v, neighborhood = node_and_neighborhood
        return G.nodes[v].get(weight, 1) / len(neighborhood - dom_set)
    vertices = set(G)
    neighborhoods = {v: {v} | set(G[v]) for v in G}
    while vertices:
        dom_node, min_set = min(neighborhoods.items(), key=_cost)
        dom_set.add(dom_node)
        del neighborhoods[dom_node]
        vertices -= min_set
    return dom_set