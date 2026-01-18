import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_bone_shaped(self):
    G = nx.DiGraph()
    G.add_node(0, demand=-4)
    G.add_node(1, demand=2)
    G.add_node(2, demand=2)
    G.add_node(3, demand=4)
    G.add_node(4, demand=-2)
    G.add_node(5, demand=-2)
    G.add_edge(0, 1, capacity=4)
    G.add_edge(0, 2, capacity=4)
    G.add_edge(4, 3, capacity=4)
    G.add_edge(5, 3, capacity=4)
    G.add_edge(0, 3, capacity=0)
    flowCost, H = nx.network_simplex(G)
    assert flowCost == 0
    assert H == {0: {1: 2, 2: 2, 3: 0}, 1: {}, 2: {}, 3: {}, 4: {3: 2}, 5: {3: 2}}
    flowCost, H = nx.capacity_scaling(G)
    assert flowCost == 0
    assert H == {0: {1: 2, 2: 2, 3: 0}, 1: {}, 2: {}, 3: {}, 4: {3: 2}, 5: {3: 2}}