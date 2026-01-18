import sage.graphs.graph as graph
import bridge_finding
def spanning_trees(G):
    """
    Function 'span' from Read paper
    """
    if G.is_connected():
        part_G = graph.Graph([])
        part_G.add_vertices(G.vertices())
        part_G.add_edges(bridge_finding.find_bridges(G))
        return rec(part_G, G)
    else:
        return []