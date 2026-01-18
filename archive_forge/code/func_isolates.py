import networkx as nx
@nx._dispatch
def isolates(G):
    """Iterator over isolates in the graph.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    iterator
        An iterator over the isolates of `G`.

    Examples
    --------
    To get a list of all isolates of a graph, use the :class:`list`
    constructor::

        >>> G = nx.Graph()
        >>> G.add_edge(1, 2)
        >>> G.add_node(3)
        >>> list(nx.isolates(G))
        [3]

    To remove all isolates in the graph, first create a list of the
    isolates, then use :meth:`Graph.remove_nodes_from`::

        >>> G.remove_nodes_from(list(nx.isolates(G)))
        >>> list(G)
        [1, 2]

    For digraphs, isolates have zero in-degree and zero out_degre::

        >>> G = nx.DiGraph([(0, 1), (1, 2)])
        >>> G.add_node(3)
        >>> list(nx.isolates(G))
        [3]

    """
    return (n for n, d in G.degree() if d == 0)