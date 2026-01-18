import networkx as nx
def small_ego_G():
    """The sample network from https://arxiv.org/pdf/1310.6753v1.pdf"""
    edges = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('b', 'e'), ('b', 'f'), ('c', 'd'), ('c', 'f'), ('c', 'h'), ('d', 'f'), ('e', 'f'), ('f', 'h'), ('h', 'j'), ('h', 'k'), ('i', 'j'), ('i', 'k'), ('j', 'k'), ('u', 'a'), ('u', 'b'), ('u', 'c'), ('u', 'd'), ('u', 'e'), ('u', 'f'), ('u', 'g'), ('u', 'h'), ('u', 'i'), ('u', 'j'), ('u', 'k')]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G