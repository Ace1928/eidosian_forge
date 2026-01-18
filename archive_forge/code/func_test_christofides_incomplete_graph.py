import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_christofides_incomplete_graph():
    G = nx.complete_graph(10)
    G.remove_edge(0, 1)
    pytest.raises(nx.NetworkXError, nx_app.christofides, G)