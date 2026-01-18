import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_effective_size_weighted_directed(self):
    D = self.D.copy()
    nx.set_edge_attributes(D, self.D_weights, 'weight')
    effective_size = nx.effective_size(D, weight='weight')
    assert effective_size[0] == pytest.approx(1.567, abs=0.001)
    assert effective_size[1] == pytest.approx(1.083, abs=0.001)
    assert effective_size[2] == pytest.approx(1, abs=0.001)