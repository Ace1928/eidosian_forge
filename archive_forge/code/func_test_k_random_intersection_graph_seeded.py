import pytest
import networkx as nx
def test_k_random_intersection_graph_seeded(self):
    G = nx.k_random_intersection_graph(10, 5, 2, seed=1234)
    assert len(G) == 10