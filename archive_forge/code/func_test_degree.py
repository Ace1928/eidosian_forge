import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_degree(self):
    assert edges_equal(self.G.degree(), list(nx.degree(self.G)))
    assert sorted(self.DG.degree()) == sorted(nx.degree(self.DG))
    assert edges_equal(self.G.degree(nbunch=[0, 1]), list(nx.degree(self.G, nbunch=[0, 1])))
    assert sorted(self.DG.degree(nbunch=[0, 1])) == sorted(nx.degree(self.DG, nbunch=[0, 1]))
    assert edges_equal(self.G.degree(weight='weight'), list(nx.degree(self.G, weight='weight')))
    assert sorted(self.DG.degree(weight='weight')) == sorted(nx.degree(self.DG, weight='weight'))