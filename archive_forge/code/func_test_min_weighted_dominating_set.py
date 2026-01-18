import pytest
import networkx as nx
from networkx.algorithms.approximation import (
def test_min_weighted_dominating_set(self):
    graph = nx.Graph()
    graph.add_edge(1, 2)
    graph.add_edge(1, 5)
    graph.add_edge(2, 3)
    graph.add_edge(2, 5)
    graph.add_edge(3, 4)
    graph.add_edge(3, 6)
    graph.add_edge(5, 6)
    vertices = {1, 2, 3, 4, 5, 6}
    dom_set = min_weighted_dominating_set(graph)
    for vertex in vertices - dom_set:
        neighbors = set(graph.neighbors(vertex))
        assert len(neighbors & dom_set) > 0, 'Non dominating set found!'