import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_graph_edit_distance_node_match(self):
    G1 = cycle_graph(5)
    G2 = cycle_graph(5)
    for n, attr in G1.nodes.items():
        attr['color'] = 'red' if n % 2 == 0 else 'blue'
    for n, attr in G2.nodes.items():
        attr['color'] = 'red' if n % 2 == 1 else 'blue'
    assert graph_edit_distance(G1, G2) == 0
    assert graph_edit_distance(G1, G2, node_match=lambda n1, n2: n1['color'] == n2['color']) == 1