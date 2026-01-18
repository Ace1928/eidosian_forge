import itertools
import pytest
import networkx as nx
def test_bad_inputs(self):
    graph = one_node_graph()
    pytest.raises(nx.NetworkXError, nx.coloring.greedy_color, graph, strategy='invalid strategy')