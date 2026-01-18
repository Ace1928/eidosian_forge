import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_minimum_cut_no_cutoff(self):
    G = self.G
    pytest.raises(nx.NetworkXError, nx.minimum_cut, G, 'x', 'y', flow_func=preflow_push, cutoff=1.0)
    pytest.raises(nx.NetworkXError, nx.minimum_cut_value, G, 'x', 'y', flow_func=preflow_push, cutoff=1.0)