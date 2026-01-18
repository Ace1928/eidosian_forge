def orient_tree(T, root):
    """Returns a digraph by orienting T outwards from root."""
    if not T.is_tree():
        raise Exception('Input must be a tree.')
    if not T.edges():
        verts = T.vertices()
        v = verts[0]
        return graph.DiGraph({v: list()})
    H = T.copy()
    H.delete_vertex(root)
    root_edges = list()
    for e in T.edges_incident(root):
        if e[0] == root:
            root_edges.append(e)
        else:
            root_edges.append((e[1], e[0], e[2]))
    pieces = list()
    for e in root_edges:
        comp = H.subgraph(H.connected_component_containing_vertex(e[1]))
        pieces.append(orient_tree(comp, e[1]))
    all_edges = root_edges
    for P in pieces:
        all_edges = all_edges + P.edges()
    return graph.DiGraph(all_edges)