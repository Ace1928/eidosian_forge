import pytest
import networkx as nx
from networkx.algorithms.approximation import k_components
from networkx.algorithms.approximation.kcomponents import _AntiGraph, _same
def test_degree_thingraph(self):
    for G, A in self.GA:
        node = list(G.nodes())[0]
        nodes = list(G.nodes())[1:4]
        assert G.degree(node) == A.degree(node)
        assert sum((d for n, d in G.degree())) == sum((d for n, d in A.degree()))
        assert sum((d for n, d in A.degree())) == sum((d for n, d in A.degree(weight='weight')))
        assert sum((d for n, d in G.degree(nodes))) == sum((d for n, d in A.degree(nodes)))