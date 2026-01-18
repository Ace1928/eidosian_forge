import networkx as nx
from networkx.exception import NetworkXAlgorithmError
from networkx.utils import not_implemented_for
@nx._dispatch(graphs='B', preserve_node_attrs=True, preserve_graph_attrs=True)
def projected_graph(B, nodes, multigraph=False):
    """Returns the projection of B onto one of its node sets.

    Returns the graph G that is the projection of the bipartite graph B
    onto the specified nodes. They retain their attributes and are connected
    in G if they have a common neighbor in B.

    Parameters
    ----------
    B : NetworkX graph
      The input graph should be bipartite.

    nodes : list or iterable
      Nodes to project onto (the "bottom" nodes).

    multigraph: bool (default=False)
       If True return a multigraph where the multiple edges represent multiple
       shared neighbors.  They edge key in the multigraph is assigned to the
       label of the neighbor.

    Returns
    -------
    Graph : NetworkX graph or multigraph
       A graph that is the projection onto the given nodes.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> B = nx.path_graph(4)
    >>> G = bipartite.projected_graph(B, [1, 3])
    >>> list(G)
    [1, 3]
    >>> list(G.edges())
    [(1, 3)]

    If nodes `a`, and `b` are connected through both nodes 1 and 2 then
    building a multigraph results in two edges in the projection onto
    [`a`, `b`]:

    >>> B = nx.Graph()
    >>> B.add_edges_from([("a", 1), ("b", 1), ("a", 2), ("b", 2)])
    >>> G = bipartite.projected_graph(B, ["a", "b"], multigraph=True)
    >>> print([sorted((u, v)) for u, v in G.edges()])
    [['a', 'b'], ['a', 'b']]

    Notes
    -----
    No attempt is made to verify that the input graph B is bipartite.
    Returns a simple graph that is the projection of the bipartite graph B
    onto the set of nodes given in list nodes.  If multigraph=True then
    a multigraph is returned with an edge for every shared neighbor.

    Directed graphs are allowed as input.  The output will also then
    be a directed graph with edges if there is a directed path between
    the nodes.

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
    overlap_weighted_projected_graph,
    generic_weighted_projected_graph
    """
    if B.is_multigraph():
        raise nx.NetworkXError('not defined for multigraphs')
    if B.is_directed():
        directed = True
        if multigraph:
            G = nx.MultiDiGraph()
        else:
            G = nx.DiGraph()
    else:
        directed = False
        if multigraph:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from(((n, B.nodes[n]) for n in nodes))
    for u in nodes:
        nbrs2 = {v for nbr in B[u] for v in B[nbr] if v != u}
        if multigraph:
            for n in nbrs2:
                if directed:
                    links = set(B[u]) & set(B.pred[n])
                else:
                    links = set(B[u]) & set(B[n])
                for l in links:
                    if not G.has_edge(u, n, l):
                        G.add_edge(u, n, key=l)
        else:
            G.add_edges_from(((u, n) for n in nbrs2))
    return G