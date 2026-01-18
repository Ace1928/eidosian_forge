import networkx as nx
from networkx.algorithms import bipartite
def test_graph_single_edge(self):
    G = nx.Graph()
    G.add_edge(0, 1)
    assert bipartite.min_edge_cover(G) == {(0, 1), (1, 0)}