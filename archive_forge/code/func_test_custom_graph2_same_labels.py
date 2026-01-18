import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph2_same_labels(self):
    G1 = nx.Graph()
    mapped = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'G', 7: 'B', 6: 'F'}
    edges1 = [(1, 2), (1, 5), (5, 6), (2, 3), (2, 4), (3, 4), (4, 5), (2, 7)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G2.remove_edge(mapped[1], mapped[2])
    G2.add_edge(mapped[1], mapped[4])
    H1 = nx.Graph(G1.subgraph([2, 3, 4, 7]))
    H2 = nx.Graph(G2.subgraph([mapped[1], mapped[4], mapped[5], mapped[6]]))
    assert vf2pp_isomorphism(H1, H2, node_label='label')
    H1.add_edges_from([(3, 7), (4, 7)])
    H2.add_edges_from([(mapped[1], mapped[6]), (mapped[4], mapped[6])])
    assert vf2pp_isomorphism(H1, H2, node_label='label')