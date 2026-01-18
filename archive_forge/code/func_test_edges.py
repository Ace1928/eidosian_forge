import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_edges(self):
    assert edges_equal(self.G.edges(), list(nx.edges(self.G)))
    assert sorted(self.DG.edges()) == sorted(nx.edges(self.DG))
    assert edges_equal(self.G.edges(nbunch=[0, 1, 3]), list(nx.edges(self.G, nbunch=[0, 1, 3])))
    assert sorted(self.DG.edges(nbunch=[0, 1, 3])) == sorted(nx.edges(self.DG, nbunch=[0, 1, 3]))