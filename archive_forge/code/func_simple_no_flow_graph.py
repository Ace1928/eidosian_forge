import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
@pytest.fixture
def simple_no_flow_graph():
    G = nx.DiGraph()
    G.add_node('s', demand=-5)
    G.add_node('t', demand=5)
    G.add_edge('s', 'a', weight=1, capacity=3)
    G.add_edge('a', 'b', weight=3)
    G.add_edge('a', 'c', weight=-6)
    G.add_edge('b', 'd', weight=1)
    G.add_edge('c', 'd', weight=-2)
    G.add_edge('d', 't', weight=1, capacity=3)
    return G