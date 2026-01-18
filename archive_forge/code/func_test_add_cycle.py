import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_add_cycle(self):
    G = self.G.copy()
    nlist = [12, 13, 14, 15]
    oklists = [[(12, 13), (12, 15), (13, 14), (14, 15)], [(12, 13), (13, 14), (14, 15), (15, 12)]]
    nx.add_cycle(G, nlist)
    assert sorted(G.edges(nlist)) in oklists
    G = self.G.copy()
    oklists = [[(12, 13, {'weight': 1.0}), (12, 15, {'weight': 1.0}), (13, 14, {'weight': 1.0}), (14, 15, {'weight': 1.0})], [(12, 13, {'weight': 1.0}), (13, 14, {'weight': 1.0}), (14, 15, {'weight': 1.0}), (15, 12, {'weight': 1.0})]]
    nx.add_cycle(G, nlist, weight=1.0)
    assert sorted(G.edges(nlist, data=True)) in oklists
    G = self.G.copy()
    nlist = [12]
    nx.add_cycle(G, nlist)
    assert nodes_equal(G, list(self.G) + nlist)
    G = self.G.copy()
    nlist = []
    nx.add_cycle(G, nlist)
    assert nodes_equal(G.nodes, self.Gnodes)
    assert edges_equal(G.edges, self.G.edges)