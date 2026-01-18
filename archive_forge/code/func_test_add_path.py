import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_add_path(self):
    G = self.G.copy()
    nlist = [12, 13, 14, 15]
    nx.add_path(G, nlist)
    assert edges_equal(G.edges(nlist), [(12, 13), (13, 14), (14, 15)])
    G = self.G.copy()
    nx.add_path(G, nlist, weight=2.0)
    assert edges_equal(G.edges(nlist, data=True), [(12, 13, {'weight': 2.0}), (13, 14, {'weight': 2.0}), (14, 15, {'weight': 2.0})])
    G = self.G.copy()
    nlist = ['node']
    nx.add_path(G, nlist)
    assert edges_equal(G.edges(nlist), [])
    assert nodes_equal(G, list(self.G) + ['node'])
    G = self.G.copy()
    nlist = iter(['node'])
    nx.add_path(G, nlist)
    assert edges_equal(G.edges(['node']), [])
    assert nodes_equal(G, list(self.G) + ['node'])
    G = self.G.copy()
    nlist = [12]
    nx.add_path(G, nlist)
    assert edges_equal(G.edges(nlist), [])
    assert nodes_equal(G, list(self.G) + [12])
    G = self.G.copy()
    nlist = iter([12])
    nx.add_path(G, nlist)
    assert edges_equal(G.edges([12]), [])
    assert nodes_equal(G, list(self.G) + [12])
    G = self.G.copy()
    nlist = []
    nx.add_path(G, nlist)
    assert edges_equal(G.edges, self.G.edges)
    assert nodes_equal(G, list(self.G))
    G = self.G.copy()
    nlist = iter([])
    nx.add_path(G, nlist)
    assert edges_equal(G.edges, self.G.edges)
    assert nodes_equal(G, list(self.G))