import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_brandes_erlebach():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 6), (3, 4), (3, 6), (4, 6), (4, 7), (5, 7), (6, 8), (6, 9), (7, 8), (7, 10), (8, 11), (9, 10), (9, 11), (10, 11)])
    for flow_func in flow_funcs:
        kwargs = {'flow_func': flow_func}
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        assert 3 == local_edge_connectivity(G, 1, 11, **kwargs), errmsg
        assert 3 == nx.edge_connectivity(G, 1, 11, **kwargs), errmsg
        assert 2 == local_node_connectivity(G, 1, 11, **kwargs), errmsg
        assert 2 == nx.node_connectivity(G, 1, 11, **kwargs), errmsg
        assert 2 == nx.edge_connectivity(G, **kwargs), errmsg
        assert 2 == nx.node_connectivity(G, **kwargs), errmsg
        if flow_func is flow.preflow_push:
            assert 3 == nx.edge_connectivity(G, 1, 11, cutoff=2, **kwargs), errmsg
        else:
            assert 2 == nx.edge_connectivity(G, 1, 11, cutoff=2, **kwargs), errmsg