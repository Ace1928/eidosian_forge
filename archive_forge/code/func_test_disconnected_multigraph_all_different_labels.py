import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_disconnected_multigraph_all_different_labels(self):
    G1 = nx.MultiGraph()
    G1.add_nodes_from(list(range(10)))
    G1.add_edges_from([(i, i) for i in range(10)])
    mapped = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped
    G1.add_edges_from([(i, i) for i in range(5, 8)] * 3)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.add_edges_from([(mapped[i], mapped[i]) for i in range(3)] * 7)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.add_edges_from([(mapped[i], mapped[i]) for i in range(5, 8)] * 3)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G1.add_edges_from([(i, i) for i in range(3)] * 7)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m