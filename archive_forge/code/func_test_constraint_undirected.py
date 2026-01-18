import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_constraint_undirected(self):
    constraint = nx.constraint(self.G)
    assert constraint['G'] == pytest.approx(0.4, abs=0.001)
    assert constraint['A'] == pytest.approx(0.595, abs=0.001)
    assert constraint['C'] == pytest.approx(1, abs=0.001)