import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_edge_missing_target():
    G = nx.path_graph(4)
    for flow_func in flow_funcs:
        pytest.raises(nx.NetworkXError, nx.edge_connectivity, G, 1, 10, flow_func=flow_func)