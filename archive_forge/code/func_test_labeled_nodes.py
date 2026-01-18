import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_labeled_nodes(self):
    g1 = nx.Graph()
    nx.add_cycle(g1, range(3))
    g1.nodes[1]['attr'] = True
    g2 = g1.copy()
    g2.add_edge(1, 3)
    ismags = iso.ISMAGS(g2, g1, node_match=lambda x, y: x == y)
    matches = ismags.subgraph_isomorphisms_iter(symmetry=True)
    expected_symmetric = [{0: 0, 1: 1, 2: 2}]
    assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric)
    matches = ismags.subgraph_isomorphisms_iter(symmetry=False)
    expected_asymmetric = [{0: 2, 1: 1, 2: 0}]
    assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric + expected_asymmetric)