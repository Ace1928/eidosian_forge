import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_neighbors(self):
    assert list(self.G.neighbors(1)) == list(nx.neighbors(self.G, 1))
    assert list(self.DG.neighbors(1)) == list(nx.neighbors(self.DG, 1))