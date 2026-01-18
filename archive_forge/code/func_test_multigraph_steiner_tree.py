import pytest
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure, steiner_tree
from networkx.utils import edges_equal
def test_multigraph_steiner_tree(self):
    G = nx.MultiGraph()
    G.add_edges_from([(1, 2, 0, {'weight': 1}), (2, 3, 0, {'weight': 999}), (2, 3, 1, {'weight': 1}), (3, 4, 0, {'weight': 1}), (3, 5, 0, {'weight': 1})])
    terminal_nodes = [2, 4, 5]
    expected_edges = [(2, 3, 1, {'weight': 1}), (3, 4, 0, {'weight': 1}), (3, 5, 0, {'weight': 1})]
    for method in self.methods:
        S = steiner_tree(G, terminal_nodes, method=method)
        assert edges_equal(S.edges(data=True, keys=True), expected_edges)