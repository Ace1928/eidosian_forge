import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_graph_single_source(self):
    G = nx.Graph(self.edges)
    G.add_edge(4, 5)
    x = list(nx.edge_bfs(G, [0]))
    x_ = [(0, 1), (0, 2), (1, 2), (1, 3)]
    assert x == x_