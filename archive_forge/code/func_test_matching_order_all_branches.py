import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_matching_order_all_branches(self):
    G1 = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)])
    G1.add_node(5)
    G2 = G1.copy()
    G1.nodes[0]['label'] = 'black'
    G1.nodes[1]['label'] = 'blue'
    G1.nodes[2]['label'] = 'blue'
    G1.nodes[3]['label'] = 'red'
    G1.nodes[4]['label'] = 'red'
    G1.nodes[5]['label'] = 'blue'
    G2.nodes[0]['label'] = 'black'
    G2.nodes[1]['label'] = 'blue'
    G2.nodes[2]['label'] = 'blue'
    G2.nodes[3]['label'] = 'red'
    G2.nodes[4]['label'] = 'red'
    G2.nodes[5]['label'] = 'blue'
    l1, l2 = (nx.get_node_attributes(G1, 'label'), nx.get_node_attributes(G2, 'label'))
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
    expected = [0, 4, 1, 3, 2, 5]
    assert _matching_order(gparams) == expected