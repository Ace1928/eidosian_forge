import pytest
import networkx as nx
def test_k5(self):
    G = nx.complete_graph(5)
    assert nx.generalized_degree(G, 0) == {3: 4}
    G.remove_edge(0, 1)
    assert nx.generalized_degree(G, 0) == {2: 3}
    assert nx.generalized_degree(G, [1, 2]) == {1: {2: 3}, 2: {2: 2, 3: 2}}
    assert nx.generalized_degree(G) == {0: {2: 3}, 1: {2: 3}, 2: {2: 2, 3: 2}, 3: {2: 2, 3: 2}, 4: {2: 2, 3: 2}}