import pytest
import networkx as nx
from networkx.algorithms.approximation import k_components
from networkx.algorithms.approximation.kcomponents import _AntiGraph, _same
def test_node_not_in_graph(self):
    for G, A in self.GA:
        node = 'non_existent_node'
        pytest.raises(nx.NetworkXError, A.neighbors, node)
        pytest.raises(nx.NetworkXError, G.neighbors, node)