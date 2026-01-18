import pytest
import networkx as nx
from networkx.algorithms import edge_dfs
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_digraph_ignore2(self):
    G = nx.DiGraph()
    nx.add_path(G, range(4))
    x = list(edge_dfs(G, [0], orientation='ignore'))
    x_ = [(0, 1, FORWARD), (1, 2, FORWARD), (2, 3, FORWARD)]
    assert x == x_