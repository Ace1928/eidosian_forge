import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph4_different_labels(self):
    G1 = nx.MultiGraph()
    edges1 = [(1, 2), (1, 2), (2, 2), (2, 3), (3, 8), (3, 8), (3, 4), (4, 5), (4, 5), (4, 5), (4, 6), (3, 6), (3, 6), (6, 6), (8, 7), (7, 7), (8, 9), (9, 9), (8, 9), (8, 9), (5, 9), (10, 11), (11, 12), (12, 13), (11, 13)]
    mapped = {1: 'n', 2: 'm', 3: 'l', 4: 'j', 5: 'k', 6: 'i', 7: 'g', 8: 'h', 9: 'f', 10: 'b', 11: 'a', 12: 'd', 13: 'e'}
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m == mapped
    G1.add_edges_from([(2, 2), (2, 3), (2, 8), (3, 4)])
    G2.add_edges_from([('m', 'm'), ('m', 'l'), ('m', 'h'), ('l', 'j')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m == mapped
    H1 = nx.MultiGraph(G1.subgraph([2, 3, 4, 6]))
    H2 = nx.MultiGraph(G2.subgraph(['m', 'l', 'j', 'i']))
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H1.remove_node(4)
    H2.remove_node('j')
    H1.remove_edges_from([(2, 2), (2, 3), (6, 6)])
    H2.remove_edges_from([('l', 'i'), ('m', 'm'), ('m', 'm')])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m
    for n1, n2 in zip(H1.nodes(), H2.nodes()):
        H1.nodes[n1]['label'] = 'red'
        H2.nodes[n2]['label'] = 'red'
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H1.remove_nodes_from([3, 6])
    H2.remove_nodes_from(['m', 'l'])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H1.remove_edge(2, 2)
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m
    H2.remove_edge('i', 'i')
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    S1 = nx.compose(H1, nx.MultiGraph(G1.subgraph([10, 11, 12, 13])))
    S2 = nx.compose(H2, nx.MultiGraph(G2.subgraph(['a', 'b', 'd', 'e'])))
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    S1.add_edges_from([(13, 13), (13, 13), (2, 13)])
    S2.add_edges_from([('a', 'a'), ('a', 'a'), ('i', 'e')])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m