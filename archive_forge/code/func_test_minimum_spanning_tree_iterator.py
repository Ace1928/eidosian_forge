import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_minimum_spanning_tree_iterator(self):
    """
        Tests that the spanning trees are correctly returned in increasing order
        """
    tree_index = 0
    for tree in nx.SpanningTreeIterator(self.G):
        actual = sorted(tree.edges(data=True))
        assert edges_equal(actual, self.spanning_trees[tree_index])
        tree_index += 1