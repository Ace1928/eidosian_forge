import os
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.utils import edges_equal
def test_intersection_attributes_node_sets_different():
    g = nx.Graph()
    g.add_node(0, x=4)
    g.add_node(1, x=5)
    g.add_node(2, x=3)
    g.add_edge(0, 1, size=5)
    g.graph['name'] = 'g'
    h = g.copy()
    h.graph['name'] = 'h'
    h.graph['attr'] = 'attr'
    h.nodes[0]['x'] = 7
    h.remove_node(2)
    gh = nx.intersection(g, h)
    assert set(gh.nodes()) == set(h.nodes())
    assert sorted(gh.edges()) == sorted(g.edges())