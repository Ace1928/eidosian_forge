import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_neg_infinite_demand_raise(simple_flow_graph):
    G = simple_flow_graph
    inf = float('inf')
    nx.set_node_attributes(G, {'a': {'demand': -inf}})
    pytest.raises(nx.NetworkXError, nx.network_simplex, G)