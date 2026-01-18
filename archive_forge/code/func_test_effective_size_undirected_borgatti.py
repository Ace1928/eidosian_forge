import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_effective_size_undirected_borgatti(self):
    effective_size = nx.effective_size(self.G)
    assert effective_size['G'] == pytest.approx(4.67, abs=0.01)
    assert effective_size['A'] == pytest.approx(2.5, abs=0.01)
    assert effective_size['C'] == pytest.approx(1, abs=0.01)