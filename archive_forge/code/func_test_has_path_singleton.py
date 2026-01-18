import pytest
import networkx as nx
def test_has_path_singleton(self):
    G = nx.empty_graph(1)
    assert nx.has_path(G, 0, 0)