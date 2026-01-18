import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_max_flow_min_cost(self):
    G = nx.DiGraph()
    G.add_edge('s', 'a', bandwidth=6)
    G.add_edge('s', 'c', bandwidth=10, cost=10)
    G.add_edge('a', 'b', cost=6)
    G.add_edge('b', 'd', bandwidth=8, cost=7)
    G.add_edge('c', 'd', cost=10)
    G.add_edge('d', 't', bandwidth=5, cost=5)
    soln = {'s': {'a': 5, 'c': 0}, 'a': {'b': 5}, 'b': {'d': 5}, 'c': {'d': 0}, 'd': {'t': 5}, 't': {}}
    flow = nx.max_flow_min_cost(G, 's', 't', capacity='bandwidth', weight='cost')
    assert flow == soln
    assert nx.cost_of_flow(G, flow, weight='cost') == 90
    G.add_edge('t', 's', cost=-100)
    flowCost, flow = nx.capacity_scaling(G, capacity='bandwidth', weight='cost')
    G.remove_edge('t', 's')
    assert flowCost == -410
    assert flow['t']['s'] == 5
    del flow['t']['s']
    assert flow == soln
    assert nx.cost_of_flow(G, flow, weight='cost') == 90