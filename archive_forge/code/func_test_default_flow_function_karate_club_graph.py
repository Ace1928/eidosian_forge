from itertools import combinations
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_default_flow_function_karate_club_graph(self):
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1, 'capacity')
    T = nx.gomory_hu_tree(G)
    assert nx.is_tree(T)
    for u, v in combinations(G, 2):
        cut_value, edge = self.minimum_edge_weight(T, u, v)
        assert nx.minimum_cut_value(G, u, v) == cut_value