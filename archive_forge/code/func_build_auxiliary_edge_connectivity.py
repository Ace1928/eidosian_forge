import networkx as nx
@nx._dispatch
def build_auxiliary_edge_connectivity(G):
    """Auxiliary digraph for computing flow based edge connectivity

    If the input graph is undirected, we replace each edge (`u`,`v`) with
    two reciprocal arcs (`u`, `v`) and (`v`, `u`) and then we set the attribute
    'capacity' for each arc to 1. If the input graph is directed we simply
    add the 'capacity' attribute. Part of algorithm 1 in [1]_ .

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms. (this is a
        chapter, look for the reference of the book).
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf
    """
    if G.is_directed():
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges(), capacity=1)
        return H
    else:
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for source, target in G.edges():
            H.add_edges_from([(source, target), (target, source)], capacity=1)
        return H