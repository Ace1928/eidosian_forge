import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_digraph_infcap_edges(self):
    G = nx.DiGraph()
    G.add_edge('s', 'a')
    G.add_edge('s', 'b', capacity=30)
    G.add_edge('a', 'c', capacity=25)
    G.add_edge('b', 'c', capacity=12)
    G.add_edge('a', 't', capacity=60)
    G.add_edge('c', 't')
    H = {'s': {'a': 85, 'b': 12}, 'a': {'c': 25, 't': 60}, 'b': {'c': 12}, 'c': {'t': 37}, 't': {}}
    compare_flows_and_cuts(G, 's', 't', H, 97)
    G = nx.DiGraph()
    G.add_edge('s', 'a', capacity=85)
    G.add_edge('s', 'b', capacity=30)
    G.add_edge('a', 'c')
    G.add_edge('c', 'a')
    G.add_edge('b', 'c', capacity=12)
    G.add_edge('a', 't', capacity=60)
    G.add_edge('c', 't', capacity=37)
    H = {'s': {'a': 85, 'b': 12}, 'a': {'c': 25, 't': 60}, 'c': {'a': 0, 't': 37}, 'b': {'c': 12}, 't': {}}
    compare_flows_and_cuts(G, 's', 't', H, 97)