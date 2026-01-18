import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_feasibility_same_labels(self):
    G1 = nx.MultiGraph([(0, 1), (0, 1), (1, 2), (1, 2), (1, 14), (0, 4), (1, 5), (2, 6), (3, 7), (3, 6), (4, 10), (4, 9), (6, 10), (20, 9), (20, 9), (20, 9), (20, 15), (20, 15), (20, 12), (20, 11), (20, 11), (20, 11), (12, 13), (11, 13), (20, 8), (20, 8), (20, 3), (20, 3), (20, 5), (20, 5), (20, 5), (20, 0), (20, 0), (20, 0)])
    mapped = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 20: 'x'}
    G2 = nx.relabel_nodes(G1, mapped)
    l1 = {n: 'blue' for n in G1.nodes()}
    l2 = {mapped[n]: 'blue' for n in G1.nodes()}
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), None)
    sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c', 3: 'd'}, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, {4, 5, 6, 7, 14}, None, {9, 10, 15, 12, 11, 13, 8}, None, {'e', 'f', 'g', 'h', 'o'}, None, {'j', 'k', 'l', 'm', 'n', 'i', 'p'}, None)
    u, v = (20, 'x')
    assert not _cut_PT(u, v, gparams, sparams)
    G2.remove_edges_from([(mapped[20], mapped[3]), (mapped[20], mapped[3])])
    G2.add_edges_from([(mapped[20], mapped[2]), (mapped[20], mapped[2])])
    assert not _cut_PT(u, v, gparams, sparams)
    assert not _consistent_PT(u, v, gparams, sparams)
    G1.remove_edges_from([(20, 3), (20, 3)])
    G1.add_edges_from([(20, 2), (20, 2)])
    assert not _cut_PT(u, v, gparams, sparams)
    assert _consistent_PT(u, v, gparams, sparams)
    G2.add_edges_from([(v, mapped[10])] * 5)
    assert _cut_PT(u, v, gparams, sparams)
    assert _consistent_PT(u, v, gparams, sparams)
    G1.add_edges_from([(u, 10)] * 5)
    assert not _cut_PT(u, v, gparams, sparams)
    assert _consistent_PT(u, v, gparams, sparams)