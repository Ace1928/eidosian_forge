import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_cut_different_labels(self):
    G1 = nx.DiGraph([(0, 1), (1, 2), (14, 1), (0, 4), (1, 5), (2, 6), (3, 7), (3, 6), (10, 4), (4, 9), (6, 10), (20, 9), (20, 15), (20, 12), (20, 11), (12, 13), (11, 13), (20, 8), (20, 3), (20, 5), (0, 20)])
    mapped = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 20: 'x'}
    G2 = nx.relabel_nodes(G1, mapped)
    l1 = {n: 'none' for n in G1.nodes()}
    l2 = {}
    l1.update({9: 'blue', 15: 'blue', 12: 'blue', 11: 'green', 3: 'green', 8: 'red', 0: 'red', 5: 'yellow'})
    l2.update({mapped[n]: l for n, l in l1.items()})
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
    sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, {4, 5, 6, 7, 20}, {14, 20}, {9, 10, 15, 12, 11, 13, 8}, None, {'e', 'f', 'g', 'x'}, {'o', 'x'}, {'j', 'k', 'l', 'm', 'n', 'i', 'p'}, None)
    u, v = (20, 'x')
    assert not _cut_PT(u, v, gparams, sparams)
    l1.update({9: 'red'})
    assert _cut_PT(u, v, gparams, sparams)
    l2.update({mapped[9]: 'red'})
    assert not _cut_PT(u, v, gparams, sparams)
    G1.add_edge(u, 4)
    assert _cut_PT(u, v, gparams, sparams)
    G2.add_edge(v, mapped[4])
    assert not _cut_PT(u, v, gparams, sparams)
    G1.add_edge(u, 14)
    assert _cut_PT(u, v, gparams, sparams)
    G2.add_edge(v, mapped[14])
    assert not _cut_PT(u, v, gparams, sparams)
    G2.remove_edge(v, mapped[8])
    assert _cut_PT(u, v, gparams, sparams)
    G1.remove_edge(u, 8)
    assert not _cut_PT(u, v, gparams, sparams)
    G1.add_edge(8, 3)
    G2.add_edge(mapped[8], mapped[3])
    sparams.T1.add(8)
    sparams.T2.add(mapped[8])
    sparams.T1_tilde.remove(8)
    sparams.T2_tilde.remove(mapped[8])
    assert not _cut_PT(u, v, gparams, sparams)
    G1.remove_node(5)
    l1.pop(5)
    sparams.T1.remove(5)
    assert _cut_PT(u, v, gparams, sparams)
    G2.remove_node(mapped[5])
    l2.pop(mapped[5])
    sparams.T2.remove(mapped[5])
    assert not _cut_PT(u, v, gparams, sparams)