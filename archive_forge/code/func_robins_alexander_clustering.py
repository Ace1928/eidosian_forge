import itertools
import networkx as nx
@nx._dispatch
def robins_alexander_clustering(G):
    """Compute the bipartite clustering of G.

    Robins and Alexander [1]_ defined bipartite clustering coefficient as
    four times the number of four cycles `C_4` divided by the number of
    three paths `L_3` in a bipartite graph:

    .. math::

       CC_4 = \\frac{4 * C_4}{L_3}

    Parameters
    ----------
    G : graph
        a bipartite graph

    Returns
    -------
    clustering : float
       The Robins and Alexander bipartite clustering for the input graph.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.davis_southern_women_graph()
    >>> print(round(bipartite.robins_alexander_clustering(G), 3))
    0.468

    See Also
    --------
    latapy_clustering
    networkx.algorithms.cluster.square_clustering

    References
    ----------
    .. [1] Robins, G. and M. Alexander (2004). Small worlds among interlocking
           directors: Network structure and distance in bipartite graphs.
           Computational & Mathematical Organization Theory 10(1), 69–94.

    """
    if G.order() < 4 or G.size() < 3:
        return 0
    L_3 = _threepaths(G)
    if L_3 == 0:
        return 0
    C_4 = _four_cycles(G)
    return 4.0 * C_4 / L_3