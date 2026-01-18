from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_multigraph_unweighted(self):
    edges = [(1, 2), (2, 3), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
    G = nx.MultiDiGraph(edges)
    assert nx.dag_longest_path_length(G) == 4