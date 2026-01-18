import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_default(self):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4], bipartite=0)
    G.add_nodes_from(['a', 'b', 'c'], bipartite=1)
    G.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])
    min_cover = bipartite.min_edge_cover(G)
    assert nx.is_edge_cover(G, min_cover)
    assert len(min_cover) == 8