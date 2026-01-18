import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph5_same_labels(self):
    G1 = nx.Graph()
    edges1 = [(1, 5), (1, 2), (1, 4), (2, 3), (2, 6), (3, 4), (3, 7), (4, 8), (5, 8), (5, 6), (6, 7), (7, 8)]
    mapped = {1: 'a', 2: 'h', 3: 'd', 4: 'i', 5: 'g', 6: 'b', 7: 'j', 8: 'c'}
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.add_edges_from([(3, 6), (2, 7), (2, 5), (1, 3), (4, 7), (6, 8)])
    G2.add_edges_from([(mapped[6], mapped[3]), (mapped[2], mapped[7]), (mapped[1], mapped[6]), (mapped[5], mapped[7]), (mapped[3], mapped[8]), (mapped[2], mapped[4])])
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    H1 = nx.Graph(G1.subgraph([1, 5, 8, 6, 7, 3]))
    H2 = nx.Graph(G2.subgraph([mapped[1], mapped[4], mapped[8], mapped[7], mapped[3], mapped[5]]))
    assert vf2pp_isomorphism(H1, H2, node_label='label')
    H1.remove_node(8)
    H2.remove_node(mapped[7])
    assert vf2pp_isomorphism(H1, H2, node_label='label')
    H1.add_edge(1, 6)
    H1.remove_edge(3, 6)
    assert vf2pp_isomorphism(H1, H2, node_label='label')