import pytest
import networkx as nx
from networkx.algorithms.approximation import k_components
from networkx.algorithms.approximation.kcomponents import _AntiGraph, _same
def test_adj(self):
    for G, A in self.GA:
        for n, nbrs in G.adj.items():
            a_adj = sorted(((n, sorted(ad)) for n, ad in A.adj.items()))
            g_adj = sorted(((n, sorted(ad)) for n, ad in G.adj.items()))
            assert a_adj == g_adj