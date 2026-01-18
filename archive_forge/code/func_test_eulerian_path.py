import collections
import pytest
import networkx as nx
def test_eulerian_path(self):
    x = [(4, 0), (0, 1), (1, 2), (2, 0)]
    for e1, e2 in zip(x, nx.eulerian_path(nx.DiGraph(x))):
        assert e1 == e2