import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def communicability_betweenness_centrality(G):
    """Returns subgraph communicability for all pairs of nodes in G.

    Communicability betweenness measure makes use of the number of walks
    connecting every pair of nodes as the basis of a betweenness centrality
    measure.

    Parameters
    ----------
    G: graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with communicability betweenness as the value.

    Raises
    ------
    NetworkXError
        If the graph is not undirected and simple.

    Notes
    -----
    Let `G=(V,E)` be a simple undirected graph with `n` nodes and `m` edges,
    and `A` denote the adjacency matrix of `G`.

    Let `G(r)=(V,E(r))` be the graph resulting from
    removing all edges connected to node `r` but not the node itself.

    The adjacency matrix for `G(r)` is `A+E(r)`,  where `E(r)` has nonzeros
    only in row and column `r`.

    The subraph betweenness of a node `r`  is [1]_

    .. math::

         \\omega_{r} = \\frac{1}{C}\\sum_{p}\\sum_{q}\\frac{G_{prq}}{G_{pq}},
         p\\neq q, q\\neq r,

    where
    `G_{prq}=(e^{A}_{pq} - (e^{A+E(r)})_{pq}`  is the number of walks
    involving node r,
    `G_{pq}=(e^{A})_{pq}` is the number of closed walks starting
    at node `p` and ending at node `q`,
    and `C=(n-1)^{2}-(n-1)` is a normalization factor equal to the
    number of terms in the sum.

    The resulting `\\omega_{r}` takes values between zero and one.
    The lower bound cannot be attained for a connected
    graph, and the upper bound is attained in the star graph.

    References
    ----------
    .. [1] Ernesto Estrada, Desmond J. Higham, Naomichi Hatano,
       "Communicability Betweenness in Complex Networks"
       Physica A 388 (2009) 764-774.
       https://arxiv.org/abs/0905.4102

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    >>> cbc = nx.communicability_betweenness_centrality(G)
    >>> print([f"{node} {cbc[node]:0.2f}" for node in sorted(cbc)])
    ['0 0.03', '1 0.45', '2 0.51', '3 0.45', '4 0.40', '5 0.19', '6 0.03']
    """
    import numpy as np
    import scipy as sp
    nodelist = list(G)
    n = len(nodelist)
    A = nx.to_numpy_array(G, nodelist)
    A[np.nonzero(A)] = 1
    expA = sp.linalg.expm(A)
    mapping = dict(zip(nodelist, range(n)))
    cbc = {}
    for v in G:
        i = mapping[v]
        row = A[i, :].copy()
        col = A[:, i].copy()
        A[i, :] = 0
        A[:, i] = 0
        B = (expA - sp.linalg.expm(A)) / expA
        B[i, :] = 0
        B[:, i] = 0
        B -= np.diag(np.diag(B))
        cbc[v] = B.sum()
        A[i, :] = row
        A[:, i] = col
    order = len(cbc)
    if order > 2:
        scale = 1.0 / ((order - 1.0) ** 2 - (order - 1.0))
        for v in cbc:
            cbc[v] *= scale
    return cbc