import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_christofides_ignore_selfloops():
    G = nx.complete_graph(5)
    G.add_edge(3, 3)
    cycle = nx_app.christofides(G)
    assert len(cycle) - 1 == len(G) == len(set(cycle))