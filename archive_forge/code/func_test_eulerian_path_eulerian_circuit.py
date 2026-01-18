import collections
import pytest
import networkx as nx
def test_eulerian_path_eulerian_circuit(self):
    G = nx.DiGraph()
    result = [(1, 2), (2, 3), (3, 4), (4, 1)]
    result2 = [(2, 3), (3, 4), (4, 1), (1, 2)]
    result3 = [(3, 4), (4, 1), (1, 2), (2, 3)]
    G.add_edges_from(result)
    assert result == list(nx.eulerian_path(G))
    assert result == list(nx.eulerian_path(G, source=1))
    assert result2 == list(nx.eulerian_path(G, source=2))
    assert result3 == list(nx.eulerian_path(G, source=3))