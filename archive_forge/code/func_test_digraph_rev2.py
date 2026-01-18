import pytest
import networkx as nx
from networkx.algorithms import edge_dfs
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_digraph_rev2(self):
    G = nx.DiGraph()
    nx.add_path(G, range(4))
    x = list(edge_dfs(G, [3], orientation='reverse'))
    x_ = [(2, 3, REVERSE), (1, 2, REVERSE), (0, 1, REVERSE)]
    assert x == x_