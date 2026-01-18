import networkx as nx
from networkx.exception import NetworkXAlgorithmError
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
@nx._dispatch(graphs='B')
def collaboration_weighted_projected_graph(B, nodes):
    """Newman's weighted projection of B onto one of its node sets.

    The collaboration weighted projection is the projection of the
    bipartite network B onto the specified nodes with weights assigned
    using Newman's collaboration model [1]_:

    .. math::

        w_{u, v} = \\sum_k \\frac{\\delta_{u}^{k} \\delta_{v}^{k}}{d_k - 1}

    where `u` and `v` are nodes from the bottom bipartite node set,
    and `k` is a node of the top node set.
    The value `d_k` is the degree of node `k` in the bipartite
    network and `\\delta_{u}^{k}` is 1 if node `u` is
    linked to node `k` in the original bipartite graph or 0 otherwise.

    The nodes retain their attributes and are connected in the resulting
    graph if have an edge to a common node in the original bipartite
    graph.

    Parameters
    ----------
    B : NetworkX graph
      The input graph should be bipartite.

    nodes : list or iterable
      Nodes to project onto (the "bottom" nodes).

    Returns
    -------
    Graph : NetworkX graph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(5)
    >>> B.add_edge(1, 5)
    >>> G = bipartite.collaboration_weighted_projected_graph(B, [0, 2, 4, 5])
    >>> list(G)
    [0, 2, 4, 5]
    >>> for edge in sorted(G.edges(data=True)):
    ...     print(edge)
    ...
    (0, 2, {'weight': 0.5})
    (0, 5, {'weight': 0.5})
    (2, 4, {'weight': 1.0})
    (2, 5, {'weight': 0.5})

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
    overlap_weighted_projected_graph,
    generic_weighted_projected_graph,
    projected_graph

    References
    ----------
    .. [1] Scientific collaboration networks: II.
        Shortest paths, weighted networks, and centrality,
        M. E. J. Newman, Phys. Rev. E 64, 016132 (2001).
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
        nbrs2 = {n for nbr in unbrs for n in B[nbr] if n != u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            common_degree = (len(B[n]) for n in unbrs & vnbrs)
            weight = sum((1.0 / (deg - 1) for deg in common_degree if deg > 1))
            G.add_edge(u, v, weight=weight)
    return G