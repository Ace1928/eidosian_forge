import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_dodecahedral():
    G = nx.dodecahedral_graph()
    for flow_func in flow_funcs:
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        assert 3 == nx.node_connectivity(G, flow_func=flow_func), errmsg
        assert 3 == nx.edge_connectivity(G, flow_func=flow_func), errmsg