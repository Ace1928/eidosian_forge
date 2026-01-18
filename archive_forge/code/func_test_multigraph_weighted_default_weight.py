from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_multigraph_weighted_default_weight(self):
    G = nx.MultiDiGraph([(1, 2), (2, 3)])
    G.add_weighted_edges_from([(1, 3, 1), (1, 3, 5), (1, 3, 2)])
    assert nx.dag_longest_path(G) == [1, 3]
    assert nx.dag_longest_path(G, default_weight=3) == [1, 2, 3]