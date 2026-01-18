import collections
import pytest
import networkx as nx
def test_is_eulerian2(self):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    assert not nx.is_eulerian(G)
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    assert not nx.is_eulerian(G)
    G = nx.MultiDiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(2, 3)
    G.add_edge(3, 1)
    assert not nx.is_eulerian(G)