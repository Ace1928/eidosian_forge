import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_negative_capacity_raise(simple_flow_graph):
    G = simple_flow_graph
    nx.set_edge_attributes(G, {('a', 'b'): {'weight': 1}, ('b', 'd'): {'capacity': -9}})
    pytest.raises(nx.NetworkXUnfeasible, nx.network_simplex, G)