import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph3_different_labels(self):
    G1 = nx.Graph()
    mapped = {1: 9, 2: 8, 3: 7, 4: 6, 5: 3, 8: 5, 9: 4, 7: 1, 6: 2}
    edges1 = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 7), (4, 9), (5, 8), (8, 9), (5, 6), (6, 7), (5, 2)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
    G1.add_edge(1, 7)
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    G2.add_edge(9, 1)
    assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
    G1.add_node('A')
    G2.add_node('K')
    G1.nodes['A']['label'] = 'green'
    G2.nodes['K']['label'] = 'green'
    mapped.update({'A': 'K'})
    assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
    G1.add_edge('A', 6)
    G2.add_edge('K', 5)
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    G1.add_edge(1, 5)
    G1.add_edge(2, 9)
    G2.add_edge(9, 3)
    G2.add_edge(8, 4)
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    for node in G1.nodes():
        color = 'red'
        G1.nodes[node]['label'] = color
        G2.nodes[mapped[node]]['label'] = color
    assert vf2pp_isomorphism(G1, G2, node_label='label')