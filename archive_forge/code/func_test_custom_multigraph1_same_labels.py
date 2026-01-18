import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph1_same_labels(self):
    G1 = nx.MultiGraph()
    mapped = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'Z', 6: 'E'}
    edges1 = [(1, 2), (1, 3), (1, 4), (1, 4), (1, 4), (2, 3), (2, 6), (2, 6), (3, 4), (3, 4), (5, 1), (5, 1), (5, 2), (5, 2)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.remove_edges_from([(2, 6), (2, 6)])
    G1.add_edges_from([(3, 6), (3, 6)])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.remove_edge(mapped[1], mapped[4])
    G1.remove_edge(1, 4)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_edges_from([(5, 5), (5, 5), (1, 1)])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.add_edges_from([(mapped[1], mapped[1]), (mapped[4], mapped[4]), (mapped[4], mapped[4])])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m