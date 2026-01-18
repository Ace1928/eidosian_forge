from functools import partial
import pytest
import networkx as nx
def test_bfs_tree_isolates(self):
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    T = nx.bfs_tree(G, source=1)
    assert sorted(T.nodes()) == [1]
    assert sorted(T.edges()) == []