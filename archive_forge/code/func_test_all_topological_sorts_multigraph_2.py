from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_all_topological_sorts_multigraph_2(self):
    N = 9
    edges = []
    for i in range(1, N):
        edges.extend([(i, i + 1)] * i)
    DG = nx.MultiDiGraph(edges)
    assert list(nx.all_topological_sorts(DG)) == [list(range(1, N + 1))]