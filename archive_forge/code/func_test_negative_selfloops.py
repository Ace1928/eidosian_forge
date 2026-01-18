import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_negative_selfloops(self):
    """Negative selfloops should cause an exception if uncapacitated and
        always be saturated otherwise.
        """
    G = nx.DiGraph()
    G.add_edge(1, 1, weight=-1)
    pytest.raises(nx.NetworkXUnbounded, nx.network_simplex, G)
    pytest.raises(nx.NetworkXUnbounded, nx.capacity_scaling, G)
    G[1][1]['capacity'] = 2
    flowCost, H = nx.network_simplex(G)
    assert flowCost == -2
    assert H == {1: {1: 2}}
    flowCost, H = nx.capacity_scaling(G)
    assert flowCost == -2
    assert H == {1: {1: 2}}
    G = nx.MultiDiGraph()
    G.add_edge(1, 1, 'x', weight=-1)
    G.add_edge(1, 1, 'y', weight=1)
    pytest.raises(nx.NetworkXUnbounded, nx.network_simplex, G)
    pytest.raises(nx.NetworkXUnbounded, nx.capacity_scaling, G)
    G[1][1]['x']['capacity'] = 2
    flowCost, H = nx.network_simplex(G)
    assert flowCost == -2
    assert H == {1: {1: {'x': 2, 'y': 0}}}
    flowCost, H = nx.capacity_scaling(G)
    assert flowCost == -2
    assert H == {1: {1: {'x': 2, 'y': 0}}}