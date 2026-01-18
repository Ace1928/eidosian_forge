import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_digraph3(self):
    G = nx.DiGraph()
    G.add_edge('s', 'v1', capacity=16.0)
    G.add_edge('s', 'v2', capacity=13.0)
    G.add_edge('v1', 'v2', capacity=10.0)
    G.add_edge('v2', 'v1', capacity=4.0)
    G.add_edge('v1', 'v3', capacity=12.0)
    G.add_edge('v3', 'v2', capacity=9.0)
    G.add_edge('v2', 'v4', capacity=14.0)
    G.add_edge('v4', 'v3', capacity=7.0)
    G.add_edge('v3', 't', capacity=20.0)
    G.add_edge('v4', 't', capacity=4.0)
    H = {'s': {'v1': 12.0, 'v2': 11.0}, 'v2': {'v1': 0, 'v4': 11.0}, 'v1': {'v2': 0, 'v3': 12.0}, 'v3': {'v2': 0, 't': 19.0}, 'v4': {'v3': 7.0, 't': 4.0}, 't': {}}
    compare_flows_and_cuts(G, 's', 't', H, 23.0)