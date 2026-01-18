import collections
import pytest
import networkx as nx
def test_has_eulerian_path_non_cyclic(self):
    assert nx.has_eulerian_path(nx.path_graph(4))
    G = nx.path_graph(6, create_using=nx.DiGraph)
    assert nx.has_eulerian_path(G)