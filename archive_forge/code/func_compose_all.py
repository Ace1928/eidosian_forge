from itertools import chain, repeat
import networkx as nx
@nx._dispatch(graphs='[graphs]', preserve_all_attrs=True)
def compose_all(graphs):
    """Returns the composition of all graphs.

    Composition is the simple union of the node sets and edge sets.
    The node sets of the supplied graphs need not be disjoint.

    Parameters
    ----------
    graphs : iterable
       Iterable of NetworkX graphs

    Returns
    -------
    C : A graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    NetworkXError
        In case of mixed type graphs, like MultiGraph and Graph, or directed and undirected graphs.

    Examples
    --------
    >>> G1 = nx.Graph([(1, 2), (2, 3)])
    >>> G2 = nx.Graph([(3, 4), (5, 6)])
    >>> C = nx.compose_all([G1, G2])
    >>> list(C.nodes())
    [1, 2, 3, 4, 5, 6]
    >>> list(C.edges())
    [(1, 2), (2, 3), (3, 4), (5, 6)]

    Notes
    -----
    For operating on mixed type graphs, they should be converted to the same type.

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.
    """
    R = None
    for i, G in enumerate(graphs):
        if i == 0:
            R = G.__class__()
        elif G.is_directed() != R.is_directed():
            raise nx.NetworkXError('All graphs must be directed or undirected.')
        elif G.is_multigraph() != R.is_multigraph():
            raise nx.NetworkXError('All graphs must be graphs or multigraphs.')
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=True))
        R.add_edges_from(G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True))
    if R is None:
        raise ValueError('cannot apply compose_all to an empty list')
    return R