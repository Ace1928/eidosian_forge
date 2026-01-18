import networkx as nx
@nx._dispatch
def build_auxiliary_node_connectivity(G):
    """Creates a directed graph D from an undirected graph G to compute flow
    based node connectivity.

    For an undirected graph G having `n` nodes and `m` edges we derive a
    directed graph D with `2n` nodes and `2m+n` arcs by replacing each
    original node `v` with two nodes `vA`, `vB` linked by an (internal)
    arc in D. Then for each edge (`u`, `v`) in G we add two arcs (`uB`, `vA`)
    and (`vB`, `uA`) in D. Finally we set the attribute capacity = 1 for each
    arc in D [1]_.

    For a directed graph having `n` nodes and `m` arcs we derive a
    directed graph D with `2n` nodes and `m+n` arcs by replacing each
    original node `v` with two nodes `vA`, `vB` linked by an (internal)
    arc (`vA`, `vB`) in D. Then for each arc (`u`, `v`) in G we add one
    arc (`uB`, `vA`) in D. Finally we set the attribute capacity = 1 for
    each arc in D.

    A dictionary with a mapping between nodes in the original graph and the
    auxiliary digraph is stored as a graph attribute: D.graph['mapping'].

    References
    ----------
    .. [1] Kammer, Frank and Hanjo Taubig. Graph Connectivity. in Brandes and
        Erlebach, 'Network Analysis: Methodological Foundations', Lecture
        Notes in Computer Science, Volume 3418, Springer-Verlag, 2005.
        https://doi.org/10.1007/978-3-540-31955-9_7

    """
    directed = G.is_directed()
    mapping = {}
    H = nx.DiGraph()
    for i, node in enumerate(G):
        mapping[node] = i
        H.add_node(f'{i}A', id=node)
        H.add_node(f'{i}B', id=node)
        H.add_edge(f'{i}A', f'{i}B', capacity=1)
    edges = []
    for source, target in G.edges():
        edges.append((f'{mapping[source]}B', f'{mapping[target]}A'))
        if not directed:
            edges.append((f'{mapping[target]}B', f'{mapping[source]}A'))
    H.add_edges_from(edges, capacity=1)
    H.graph['mapping'] = mapping
    return H