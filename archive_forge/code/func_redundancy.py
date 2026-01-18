import networkx as nx
def redundancy(G, u, v, weight=None):
    nmw = normalized_mutual_weight
    r = sum((nmw(G, u, w, weight=weight) * nmw(G, v, w, norm=max, weight=weight) for w in set(nx.all_neighbors(G, u))))
    return 1 - r