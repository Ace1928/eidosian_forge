import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_different_degree_sequences1(self):
    G1 = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (0, 4)])
    G2 = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (0, 4), (2, 5)])
    assert not vf2pp_is_isomorphic(G1, G2)
    G2.remove_node(3)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(['a']))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle('a'))), 'label')
    assert vf2pp_is_isomorphic(G1, G2)