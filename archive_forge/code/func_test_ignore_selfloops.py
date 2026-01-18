import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_ignore_selfloops(self):
    G = nx.complete_graph(5)
    G.add_edge(3, 3)
    cycle = self.tsp(G, 'greedy')
    assert len(cycle) - 1 == len(G) == len(set(cycle))