import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_is_frozen(self):
    assert not nx.is_frozen(self.G)
    G = nx.freeze(self.G)
    assert G.frozen == nx.is_frozen(self.G)
    assert G.frozen