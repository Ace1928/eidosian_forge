import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_digraph_infcap_path(self):
    G = nx.DiGraph()
    G.add_edge('s', 'a')
    G.add_edge('s', 'b', capacity=30)
    G.add_edge('a', 'c')
    G.add_edge('b', 'c', capacity=12)
    G.add_edge('a', 't', capacity=60)
    G.add_edge('c', 't')
    for flow_func in all_funcs:
        pytest.raises(nx.NetworkXUnbounded, flow_func, G, 's', 't')