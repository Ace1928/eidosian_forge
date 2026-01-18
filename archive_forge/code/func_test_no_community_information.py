import math
from functools import partial
import pytest
import networkx as nx
def test_no_community_information(self):
    G = nx.complete_graph(5)
    assert pytest.raises(nx.NetworkXAlgorithmError, list, self.func(G, [(0, 1)]))