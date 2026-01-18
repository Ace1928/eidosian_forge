import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_const_fail_cases(self):
    G1 = nx.DiGraph([(0, 1), (2, 1), (10, 0), (10, 3), (10, 4), (5, 10), (10, 6), (1, 4), (5, 3)])
    G2 = nx.DiGraph([('a', 'b'), ('c', 'b'), ('k', 'a'), ('k', 'd'), ('k', 'e'), ('f', 'k'), ('k', 'g'), ('b', 'e'), ('f', 'd')])
    gparams = _GraphParameters(G1, G2, None, None, None, None, None)
    sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, None, None, None, None, None, None, None, None)
    u, v = (10, 'k')
    assert _consistent_PT(u, v, gparams, sparams)
    G1.remove_node(6)
    assert _consistent_PT(u, v, gparams, sparams)
    G1.add_edge(u, 2)
    assert not _consistent_PT(u, v, gparams, sparams)
    G2.add_edge(v, 'c')
    assert _consistent_PT(u, v, gparams, sparams)
    G2.add_edge(v, 'x')
    G1.add_node(7)
    sparams.mapping.update({7: 'x'})
    sparams.reverse_mapping.update({'x': 7})
    assert not _consistent_PT(u, v, gparams, sparams)
    G1.add_edge(u, 7)
    assert _consistent_PT(u, v, gparams, sparams)