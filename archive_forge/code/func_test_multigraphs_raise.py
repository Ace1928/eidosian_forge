import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_multigraphs_raise(self):
    G = nx.MultiGraph()
    M = nx.MultiDiGraph()
    G.add_edges_from([(0, 1), (1, 0)], capacity=True)
    for flow_func in all_funcs:
        pytest.raises(nx.NetworkXError, flow_func, G, 0, 0)