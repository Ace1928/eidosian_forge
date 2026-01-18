import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_digraph5(self):
    G = nx.DiGraph()
    G.add_edge('s', 'a', capacity=2)
    G.add_edge('s', 'b', capacity=2)
    G.add_edge('a', 'b', capacity=5)
    G.add_edge('a', 't', capacity=1)
    G.add_edge('b', 'a', capacity=1)
    G.add_edge('b', 't', capacity=3)
    flowSoln = {'a': {'b': 1, 't': 1}, 'b': {'a': 0, 't': 3}, 's': {'a': 2, 'b': 2}, 't': {}}
    compare_flows_and_cuts(G, 's', 't', flowSoln, 4)