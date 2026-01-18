from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(edge_attrs='weight')
def clustering(G, nodes=None, weight=None):
    """Compute the clustering coefficient for nodes.

    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,

    .. math::

      c_u = \\frac{2 T(u)}{deg(u)(deg(u)-1)},

    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.

    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,

    .. math::

       c_u = \\frac{1}{deg(u)(deg(u)-1))}
             \\sum_{vw} (\\hat{w}_{uv} \\hat{w}_{uw} \\hat{w}_{vw})^{1/3}.

    The edge weights :math:`\\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\\hat{w}_{uv} = w_{uv}/\\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    Additionally, this weighted definition has been generalized to support negative edge weights [3]_.

    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [4]_.

    .. math::

       c_u = \\frac{T(u)}{2(deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\\leftrightarrow}(u))},

    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.


    Parameters
    ----------
    G : graph

    nodes : node, iterable of nodes, or None (default=None)
        If a singleton node, return the number of triangles for that node.
        If an iterable, compute the number of triangles for each of those nodes.
        If `None` (the default) compute the number of triangles for all nodes in `G`.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.clustering(G, 0))
    1.0
    >>> print(nx.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [3] Generalization of Clustering Coefficients to Signed Correlation Networks
       by G. Costantini and M. Perugini, PloS one, 9(2), e88669 (2014).
    .. [4] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).
    """
    if G.is_directed():
        if weight is not None:
            td_iter = _directed_weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2) for v, dt, db, t in td_iter}
        else:
            td_iter = _directed_triangles_and_degree_iter(G, nodes)
            clusterc = {v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2) for v, dt, db, t in td_iter}
    elif weight is not None:
        td_iter = _weighted_triangles_and_degree_iter(G, nodes, weight)
        clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t in td_iter}
    else:
        td_iter = _triangles_and_degree_iter(G, nodes)
        clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t, _ in td_iter}
    if nodes in G:
        return clusterc[nodes]
    return clusterc