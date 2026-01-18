import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_minimum_weight_full_matching_incomplete_graph(self):
    B = nx.Graph()
    B.add_nodes_from([1, 2], bipartite=0)
    B.add_nodes_from([3, 4], bipartite=1)
    B.add_edge(1, 4, weight=100)
    B.add_edge(2, 3, weight=100)
    B.add_edge(2, 4, weight=50)
    matching = minimum_weight_full_matching(B)
    assert matching == {1: 4, 2: 3, 4: 1, 3: 2}