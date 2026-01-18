import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_cut_same_labels(self):
    G1 = nx.DiGraph([(0, 1), (2, 1), (10, 0), (10, 3), (10, 4), (5, 10), (10, 6), (1, 4), (5, 3)])
    mapped = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 10: 'k'}
    G2 = nx.relabel_nodes(G1, mapped)
    l1 = {n: 'blue' for n in G1.nodes()}
    l2 = {n: 'blue' for n in G2.nodes()}
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
    sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, {4}, {5, 10}, {6}, None, {'e'}, {'f', 'k'}, {'g'}, None)
    u, v = (10, 'k')
    assert not _cut_PT(u, v, gparams, sparams)
    G1.remove_edge(u, 4)
    assert _cut_PT(u, v, gparams, sparams)
    G2.remove_edge(v, mapped[4])
    assert not _cut_PT(u, v, gparams, sparams)
    G1.remove_edge(5, u)
    assert _cut_PT(u, v, gparams, sparams)
    G2.remove_edge(mapped[5], v)
    assert not _cut_PT(u, v, gparams, sparams)
    G2.remove_edge(v, mapped[6])
    assert _cut_PT(u, v, gparams, sparams)
    G1.remove_edge(u, 6)
    assert not _cut_PT(u, v, gparams, sparams)
    G1.add_nodes_from([6, 7, 8])
    G2.add_nodes_from(['g', 'y', 'z'])
    sparams.T1_tilde.update({6, 7, 8})
    sparams.T2_tilde.update({'g', 'y', 'z'})
    l1 = {n: 'blue' for n in G1.nodes()}
    l2 = {n: 'blue' for n in G2.nodes()}
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
    assert not _cut_PT(u, v, gparams, sparams)