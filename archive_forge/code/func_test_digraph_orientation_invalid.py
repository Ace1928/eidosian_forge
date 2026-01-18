import pytest
import networkx as nx
from networkx.algorithms import edge_dfs
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_digraph_orientation_invalid(self):
    G = nx.DiGraph(self.edges)
    edge_iterator = edge_dfs(G, self.nodes, orientation='hello')
    pytest.raises(nx.NetworkXError, list, edge_iterator)