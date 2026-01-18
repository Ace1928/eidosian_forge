import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_icosahedral():
    G = nx.icosahedral_graph()
    for flow_func in flow_funcs:
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        assert 5 == nx.node_connectivity(G, flow_func=flow_func), errmsg
        assert 5 == nx.edge_connectivity(G, flow_func=flow_func), errmsg