import collections
import pytest
import networkx as nx
def test_has_eulerian_path_directed_graph(self):
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    assert not nx.has_eulerian_path(G)
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    assert nx.has_eulerian_path(G)
    G.add_node(3)
    assert not nx.has_eulerian_path(G)