import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph4_same_labels(self):
    G1 = nx.Graph()
    edges1 = [(1, 2), (2, 3), (3, 8), (3, 4), (4, 5), (4, 6), (3, 6), (8, 7), (8, 9), (5, 9), (10, 11), (11, 12), (12, 13), (11, 13)]
    mapped = {1: 'n', 2: 'm', 3: 'l', 4: 'j', 5: 'k', 6: 'i', 7: 'g', 8: 'h', 9: 'f', 10: 'b', 11: 'a', 12: 'd', 13: 'e'}
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.add_node(0)
    G2.add_node('z')
    G1.nodes[0]['label'] = 'green'
    G2.nodes['z']['label'] = 'blue'
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    G2.nodes['z']['label'] = 'green'
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.add_edge(2, 5)
    G2.remove_edge('i', 'l')
    G2.add_edge('g', 'l')
    G2.add_edge('m', 'f')
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.remove_node(13)
    G2.remove_node('d')
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.add_edge(0, 10)
    G2.add_edge('e', 'z')
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.add_edge(11, 3)
    G1.add_edge(0, 8)
    G2.add_edge('a', 'l')
    G2.add_edge('z', 'j')
    assert vf2pp_isomorphism(G1, G2, node_label='label')