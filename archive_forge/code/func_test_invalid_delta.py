import math
from functools import partial
import pytest
import networkx as nx
def test_invalid_delta(self):
    G = nx.complete_graph(3)
    G.add_nodes_from([0, 1, 2], community=0)
    assert pytest.raises(nx.NetworkXAlgorithmError, self.func, G, [(0, 1)], 0)
    assert pytest.raises(nx.NetworkXAlgorithmError, self.func, G, [(0, 1)], -0.5)