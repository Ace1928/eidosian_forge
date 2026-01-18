import pytest
import networkx as nx
def test_atlas(self):
    for graph in self.GAG:
        deg = (d for n, d in graph.degree())
        assert nx.is_graphical(deg, method='eg')
        assert nx.is_graphical(deg, method='hh')