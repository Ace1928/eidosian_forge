import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_multigraph_keys_tree_max(self):
    G = nx.MultiGraph()
    G.add_edge(0, 1, key='a', weight=2)
    G.add_edge(0, 1, key='b', weight=1)
    T = nx.maximum_spanning_tree(G, algorithm=self.algo)
    assert edges_equal([(0, 1, 2)], list(T.edges(data='weight')))