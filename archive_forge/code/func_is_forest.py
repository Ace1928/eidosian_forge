import networkx as nx
@nx._dispatch
def is_forest(G):
    """
    Returns True if `G` is a forest.

    A forest is a graph with no undirected cycles.

    For directed graphs, `G` is a forest if the underlying graph is a forest.
    The underlying graph is obtained by treating each directed edge as a single
    undirected edge in a multigraph.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a forest.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is empty.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
    >>> nx.is_forest(G)
    True
    >>> G.add_edge(4, 1)
    >>> nx.is_forest(G)
    False

    Notes
    -----
    In another convention, a directed forest is known as a *polyforest* and
    then *forest* corresponds to a *branching*.

    See Also
    --------
    is_branching

    """
    if len(G) == 0:
        raise nx.exception.NetworkXPointlessConcept('G has no nodes.')
    if G.is_directed():
        components = (G.subgraph(c) for c in nx.weakly_connected_components(G))
    else:
        components = (G.subgraph(c) for c in nx.connected_components(G))
    return all((len(c) - 1 == c.number_of_edges() for c in components))