import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_generate_random_paths_unweighted(self):
    np.random.seed(42)
    index_map = {}
    num_paths = 10
    path_length = 2
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(1, 2)
    G.add_edge(2, 4)
    paths = nx.generate_random_paths(G, num_paths, path_length=path_length, index_map=index_map)
    expected_paths = [[3, 0, 3], [4, 2, 1], [2, 1, 0], [2, 0, 3], [3, 0, 1], [3, 0, 1], [4, 2, 0], [2, 1, 0], [3, 0, 2], [2, 1, 2]]
    expected_map = {0: {0, 2, 3, 4, 5, 6, 7, 8}, 1: {1, 2, 4, 5, 7, 9}, 2: {1, 2, 3, 6, 7, 8, 9}, 3: {0, 3, 4, 5, 8}, 4: {1, 6}}
    assert expected_paths == list(paths)
    assert expected_map == index_map