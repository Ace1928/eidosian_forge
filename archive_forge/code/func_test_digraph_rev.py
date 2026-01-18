import pytest
import networkx as nx
from networkx.algorithms import edge_dfs
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_digraph_rev(self):
    G = nx.DiGraph(self.edges)
    x = list(edge_dfs(G, self.nodes, orientation='reverse'))
    x_ = [(1, 0, REVERSE), (0, 1, REVERSE), (2, 1, REVERSE), (3, 1, REVERSE)]
    assert x == x_