import networkx as nx
from networkx.exception import NetworkXAlgorithmError
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
@nx._dispatch(graphs='B')
def overlap_weighted_projected_graph(B, nodes, jaccard=True):
    """Overlap weighted projection of B onto one of its node sets.

    The overlap weighted projection is the projection of the bipartite
    network B onto the specified nodes with weights representing
    the Jaccard index between the neighborhoods of the two nodes in the
    original bipartite network [1]_:

    .. math::

        w_{v, u} = \\frac{|N(u) \\cap N(v)|}{|N(u) \\cup N(v)|}

    or if the parameter 'jaccard' is False, the fraction of common
    neighbors by minimum of both nodes degree in the original
    bipartite graph [1]_:

    .. math::

        w_{v, u} = \\frac{|N(u) \\cap N(v)|}{min(|N(u)|, |N(v)|)}

    The nodes retain their attributes and are connected in the resulting
    graph if have an edge to a common node in the original bipartite graph.

    Parameters
    ----------
    B : NetworkX graph
        The input graph should be bipartite.

    nodes : list or iterable
        Nodes to project onto (the "bottom" nodes).

    jaccard: Bool (default=True)

    Returns
    -------
    Graph : NetworkX graph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(5)
    >>> nodes = [0, 2, 4]
    >>> G = bipartite.overlap_weighted_projected_graph(B, nodes)
    >>> list(G)
    [0, 2, 4]
    >>> list(G.edges(data=True))
    [(0, 2, {'weight': 0.5}), (2, 4, {'weight': 0.5})]
    >>> G = bipartite.overlap_weighted_projected_graph(B, nodes, jaccard=False)
    >>> list(G.edges(data=True))
    [(0, 2, {'weight': 1.0}), (2, 4, {'weight': 1.0})]

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite.
    The graph and node properties are (shallow) copied to the projected graph.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    is_bipartite,
    is_bipartite_node_set,
    sets,
    weighted_projected_graph,
    collaboration_weighted_projected_graph,
    generic_weighted_projected_graph,
    projected_graph

    References
    ----------
    .. [1] Borgatti, S.P. and Halgin, D. In press. Analyzing Affiliation
        Networks. In Carrington, P. and Scott, J. (eds) The Sage Handbook
        of Social Network Analysis. Sage Publications.

    """
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from(((n, B.nodes[n]) for n in nodes))
    for u in nodes:
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr]} - {u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            if jaccard:
                wt = len(unbrs & vnbrs) / len(unbrs | vnbrs)
            else:
                wt = len(unbrs & vnbrs) / min(len(unbrs), len(vnbrs))
            G.add_edge(u, v, weight=wt)
    return G