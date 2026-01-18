import sage.graphs.graph as graph
import bridge_finding
def get_B(part_G, G):
    """
    Returns the set of edges not in part_G joining vertices already connected in part_G
    """
    B = []
    comps = part_G.connected_components()
    vc_dict = dict()
    for i in range(len(comps)):
        for v in comps[i]:
            vc_dict[v] = i
    X = G.edges()
    for y in part_G.edges():
        X.remove(y)
    for e in X:
        if vc_dict[e[0]] == vc_dict[e[1]]:
            B.append(e)
    return B