import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_maximum_tree(self):
    T = nx.maximum_spanning_tree(self.G, algorithm=self.algo)
    actual = sorted(T.edges(data=True))
    assert edges_equal(actual, self.maximum_spanning_edgelist)