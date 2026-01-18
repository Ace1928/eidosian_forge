import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_non_neighbors(self):
    graph = nx.complete_graph(100)
    pop = random.sample(list(graph), 1)
    nbors = list(nx.non_neighbors(graph, pop[0]))
    assert len(nbors) == 0
    graph = nx.path_graph(100)
    node = random.sample(list(graph), 1)[0]
    nbors = list(nx.non_neighbors(graph, node))
    if node != 0 and node != 99:
        assert len(nbors) == 97
    else:
        assert len(nbors) == 98
    graph = nx.star_graph(99)
    nbors = list(nx.non_neighbors(graph, 0))
    assert len(nbors) == 0
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    nbors = list(nx.non_neighbors(graph, 0))
    assert len(nbors) == 9