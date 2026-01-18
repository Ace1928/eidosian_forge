import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_is_directed(self):
    assert self.G.is_directed() == nx.is_directed(self.G)
    assert self.DG.is_directed() == nx.is_directed(self.DG)