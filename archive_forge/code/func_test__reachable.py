from itertools import combinations
import pytest
import networkx as nx
def test__reachable(large_collider_graph):
    reachable = nx.algorithms.d_separation._reachable
    g = large_collider_graph
    x = {'F', 'D'}
    ancestors = {'A', 'B', 'C', 'D', 'F'}
    assert reachable(g, x, ancestors, {'B'}) == {'B', 'F', 'D'}
    assert reachable(g, x, ancestors, set()) == ancestors