import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph3_same_labels(self):
    G1 = nx.MultiGraph()
    mapped = {1: 9, 2: 8, 3: 7, 4: 6, 5: 3, 8: 5, 9: 4, 7: 1, 6: 2}
    edges1 = [(1, 2), (1, 3), (1, 3), (2, 3), (2, 3), (3, 4), (4, 5), (4, 7), (4, 9), (4, 9), (4, 9), (5, 8), (5, 8), (8, 9), (8, 9), (5, 6), (6, 7), (6, 7), (6, 7), (5, 2)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_edges_from([(6, 9), (7, 8), (5, 8), (4, 9), (4, 9)])
    G2.add_edges_from([(mapped[6], mapped[8]), (mapped[7], mapped[9]), (mapped[5], mapped[8]), (mapped[4], mapped[9]), (mapped[4], mapped[9])])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G1.add_edges_from([(6, 8), (6, 8), (7, 9), (7, 9), (7, 9)])
    G2.add_edges_from([(mapped[6], mapped[8]), (mapped[6], mapped[9]), (mapped[7], mapped[8]), (mapped[7], mapped[9]), (mapped[7], mapped[9])])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_edges_from([(2, 7), (2, 7), (3, 6), (3, 6)])
    G2.add_edges_from([(mapped[2], mapped[7]), (mapped[2], mapped[7]), (mapped[3], mapped[6]), (mapped[3], mapped[6])])
    G1.add_node(10)
    G2.add_node('Z')
    G1.nodes[10]['label'] = 'blue'
    G2.nodes['Z']['label'] = 'blue'
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_edges_from([(10, 1), (10, 5), (10, 8), (10, 10), (10, 10)])
    G2.add_edges_from([('Z', mapped[1]), ('Z', mapped[4]), ('Z', mapped[9]), ('Z', 'Z'), ('Z', 'Z')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G1.remove_edges_from([(1, 3), (4, 9), (4, 9), (7, 9)])
    G2.remove_edges_from([(mapped[1], mapped[3]), (mapped[4], mapped[9]), (mapped[4], mapped[9]), (mapped[7], mapped[9])])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    H1 = nx.Graph(G1.subgraph([2, 3, 4, 5, 6, 7, 10]))
    H2 = nx.Graph(G2.subgraph([mapped[4], mapped[5], mapped[6], mapped[7], mapped[8], mapped[9], 'Z']))
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m
    H1.add_edges_from([(10, 2), (10, 6), (3, 6), (2, 7), (2, 6), (3, 7)])
    H2.add_edges_from([('Z', mapped[7]), (mapped[6], mapped[9]), (mapped[7], mapped[8])])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H2.remove_edge('Z', 'Z')
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m
    H1.remove_edge(10, 10)
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m