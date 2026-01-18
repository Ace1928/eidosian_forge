import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def communicability(G):
    """Returns communicability between all pairs of nodes in G.

    The communicability between pairs of nodes in G is the sum of
    walks of different lengths starting at node u and ending at node v.

    Parameters
    ----------
    G: graph

    Returns
    -------
    comm: dictionary of dictionaries
        Dictionary of dictionaries keyed by nodes with communicability
        as the value.

    Raises
    ------
    NetworkXError
       If the graph is not undirected and simple.

    See Also
    --------
    communicability_exp:
       Communicability between all pairs of nodes in G  using spectral
       decomposition.
    communicability_betweenness_centrality:
       Communicability betweenness centrality for each node in G.

    Notes
    -----
    This algorithm uses a spectral decomposition of the adjacency matrix.
    Let G=(V,E) be a simple undirected graph.  Using the connection between
    the powers  of the adjacency matrix and the number of walks in the graph,
    the communicability  between nodes `u` and `v` based on the graph spectrum
    is [1]_

    .. math::
        C(u,v)=\\sum_{j=1}^{n}\\phi_{j}(u)\\phi_{j}(v)e^{\\lambda_{j}},

    where `\\phi_{j}(u)` is the `u\\rm{th}` element of the `j\\rm{th}` orthonormal
    eigenvector of the adjacency matrix associated with the eigenvalue
    `\\lambda_{j}`.

    References
    ----------
    .. [1] Ernesto Estrada, Naomichi Hatano,
       "Communicability in complex networks",
       Phys. Rev. E 77, 036111 (2008).
       https://arxiv.org/abs/0707.0756

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    >>> c = nx.communicability(G)
    """
    import numpy as np
    nodelist = list(G)
    A = nx.to_numpy_array(G, nodelist)
    A[A != 0.0] = 1
    w, vec = np.linalg.eigh(A)
    expw = np.exp(w)
    mapping = dict(zip(nodelist, range(len(nodelist))))
    c = {}
    for u in G:
        c[u] = {}
        for v in G:
            s = 0
            p = mapping[u]
            q = mapping[v]
            for j in range(len(nodelist)):
                s += vec[:, j][p] * vec[:, j][q] * expw[j]
            c[u][v] = float(s)
    return c