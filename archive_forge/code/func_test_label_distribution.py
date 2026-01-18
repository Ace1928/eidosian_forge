import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_label_distribution(self):
    G1 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (2, 5), (2, 6)])
    G2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (2, 5), (2, 6)])
    colors1 = ['blue', 'blue', 'blue', 'yellow', 'black', 'purple', 'purple']
    colors2 = ['blue', 'blue', 'yellow', 'yellow', 'black', 'purple', 'purple']
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(colors1[::-1]))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(colors2[::-1]))), 'label')
    assert not vf2pp_is_isomorphic(G1, G2, node_label='label')
    G2.nodes[3]['label'] = 'blue'
    assert vf2pp_is_isomorphic(G1, G2, node_label='label')