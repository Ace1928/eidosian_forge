import networkx as nx
def edge_id(edge):
    return (frozenset(edge[:2]),) + edge[2:]