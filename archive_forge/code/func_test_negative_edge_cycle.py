import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_edge_cycle(self):
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    assert not nx.negative_edge_cycle(G)
    G.add_edge(8, 9, weight=-7)
    G.add_edge(9, 8, weight=3)
    graph_size = len(G)
    assert nx.negative_edge_cycle(G)
    assert graph_size == len(G)
    pytest.raises(ValueError, nx.single_source_dijkstra_path_length, G, 8)
    pytest.raises(ValueError, nx.single_source_dijkstra, G, 8)
    pytest.raises(ValueError, nx.dijkstra_predecessor_and_distance, G, 8)
    G.add_edge(9, 10)
    pytest.raises(ValueError, nx.bidirectional_dijkstra, G, 8, 10)
    G = nx.MultiDiGraph()
    G.add_edge(2, 2, weight=-1)
    assert nx.negative_edge_cycle(G)