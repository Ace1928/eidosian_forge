import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph1_different_labels(self):
    G1 = nx.Graph()
    mapped = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'Z', 6: 'E'}
    edges1 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 6), (3, 4), (5, 1), (5, 2)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped