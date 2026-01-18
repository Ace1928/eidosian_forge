from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def square_clustering(G, nodes=None):
    """Compute the squares clustering coefficient for nodes.

    For each node return the fraction of possible squares that exist at
    the node [1]_

    .. math::
       C_4(v) = \\frac{ \\sum_{u=1}^{k_v}
       \\sum_{w=u+1}^{k_v} q_v(u,w) }{ \\sum_{u=1}^{k_v}
       \\sum_{w=u+1}^{k_v} [a_v(u,w) + q_v(u,w)]},

    where :math:`q_v(u,w)` are the number of common neighbors of :math:`u` and
    :math:`w` other than :math:`v` (ie squares), and :math:`a_v(u,w) = (k_u -
    (1+q_v(u,w)+\\theta_{uv})) + (k_w - (1+q_v(u,w)+\\theta_{uw}))`, where
    :math:`\\theta_{uw} = 1` if :math:`u` and :math:`w` are connected and 0
    otherwise. [2]_

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.

    Returns
    -------
    c4 : dictionary
       A dictionary keyed by node with the square clustering coefficient value.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.square_clustering(G, 0))
    1.0
    >>> print(nx.square_clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    While :math:`C_3(v)` (triangle clustering) gives the probability that
    two neighbors of node v are connected with each other, :math:`C_4(v)` is
    the probability that two neighbors of node v share a common
    neighbor different from v. This algorithm can be applied to both
    bipartite and unipartite networks.

    References
    ----------
    .. [1] Pedro G. Lind, Marta C. González, and Hans J. Herrmann. 2005
        Cycles and clustering in bipartite networks.
        Physical Review E (72) 056127.
    .. [2] Zhang, Peng et al. Clustering Coefficient and Community Structure of
        Bipartite Networks. Physica A: Statistical Mechanics and its Applications 387.27 (2008): 6869–6875.
        https://arxiv.org/abs/0710.0117v1
    """
    if nodes is None:
        node_iter = G
    else:
        node_iter = G.nbunch_iter(nodes)
    clustering = {}
    for v in node_iter:
        clustering[v] = 0
        potential = 0
        for u, w in combinations(G[v], 2):
            squares = len((set(G[u]) & set(G[w])) - {v})
            clustering[v] += squares
            degm = squares + 1
            if w in G[u]:
                degm += 1
            potential += len(G[u]) - degm + (len(G[w]) - degm) + squares
        if potential > 0:
            clustering[v] /= potential
    if nodes in G:
        return clustering[nodes]
    return clustering