import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_infinite_weight_raise(simple_flow_graph):
    G = simple_flow_graph
    inf = float('inf')
    nx.set_edge_attributes(G, {('a', 'b'): {'weight': inf}, ('b', 'd'): {'weight': inf}})
    pytest.raises(nx.NetworkXError, nx.network_simplex, G)