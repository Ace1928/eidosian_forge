import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_find_cliques1(self):
    cl = list(nx.find_cliques(self.G))
    rcl = nx.find_cliques_recursive(self.G)
    expected = [[2, 6, 1, 3], [2, 6, 4], [5, 4, 7], [8, 9], [10, 11]]
    assert sorted(map(sorted, cl)) == sorted(map(sorted, rcl))
    assert sorted(map(sorted, cl)) == sorted(map(sorted, expected))