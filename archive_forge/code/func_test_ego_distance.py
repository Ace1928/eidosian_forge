import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_ego_distance(self):
    G = nx.Graph()
    G.add_edge(0, 1, weight=2, distance=1)
    G.add_edge(1, 2, weight=2, distance=2)
    G.add_edge(2, 3, weight=2, distance=1)
    assert nodes_equal(nx.ego_graph(G, 0, radius=3).nodes(), [0, 1, 2, 3])
    eg = nx.ego_graph(G, 0, radius=3, distance='weight')
    assert nodes_equal(eg.nodes(), [0, 1])
    eg = nx.ego_graph(G, 0, radius=3, distance='weight', undirected=True)
    assert nodes_equal(eg.nodes(), [0, 1])
    eg = nx.ego_graph(G, 0, radius=3, distance='distance')
    assert nodes_equal(eg.nodes(), [0, 1, 2])