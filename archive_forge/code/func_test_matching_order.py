import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_matching_order(self):
    labels = ['blue', 'blue', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'yellow', 'purple', 'purple', 'blue', 'blue']
    G1 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 5), (2, 4), (1, 3), (1, 4), (3, 6), (4, 6), (6, 7), (7, 8), (9, 10), (9, 11), (11, 12), (11, 13), (12, 13), (10, 13)])
    G2 = G1.copy()
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels))), 'label')
    l1, l2 = (nx.get_node_attributes(G1, 'label'), nx.get_node_attributes(G2, 'label'))
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
    expected = [9, 11, 10, 13, 12, 1, 2, 4, 0, 3, 6, 5, 7, 8]
    assert _matching_order(gparams) == expected