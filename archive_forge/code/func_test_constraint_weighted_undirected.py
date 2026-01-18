import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_constraint_weighted_undirected(self):
    G = self.G.copy()
    nx.set_edge_attributes(G, self.G_weights, 'weight')
    constraint = nx.constraint(G, weight='weight')
    assert constraint['G'] == pytest.approx(0.299, abs=0.001)
    assert constraint['A'] == pytest.approx(0.795, abs=0.001)
    assert constraint['C'] == pytest.approx(1, abs=0.001)