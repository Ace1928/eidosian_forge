import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_complete_graph_cutoff(self):
    G = nx.complete_graph(5)
    nx.set_edge_attributes(G, {(u, v): 1 for u, v in G.edges()}, 'capacity')
    for flow_func in [shortest_augmenting_path, edmonds_karp, dinitz, boykov_kolmogorov]:
        for cutoff in [3, 2, 1]:
            result = nx.maximum_flow_value(G, 0, 4, flow_func=flow_func, cutoff=cutoff)
            assert cutoff == result, f'cutoff error in {flow_func.__name__}'