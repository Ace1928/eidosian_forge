import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_effective_size_borgatti_isolated(self):
    G = self.G.copy()
    G.add_node(1)
    effective_size = nx.effective_size(G)
    assert math.isnan(effective_size[1])