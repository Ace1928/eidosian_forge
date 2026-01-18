import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_graph_infcap_edges(self):
    G = nx.Graph()
    G.add_edge('s', 'a')
    G.add_edge('s', 'b', capacity=30)
    G.add_edge('a', 'c', capacity=25)
    G.add_edge('b', 'c', capacity=12)
    G.add_edge('a', 't', capacity=60)
    G.add_edge('c', 't')
    H = {'s': {'a': 85, 'b': 12}, 'a': {'c': 25, 's': 85, 't': 60}, 'b': {'c': 12, 's': 12}, 'c': {'a': 25, 'b': 12, 't': 37}, 't': {'a': 60, 'c': 37}}
    compare_flows_and_cuts(G, 's', 't', H, 97)