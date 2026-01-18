from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_all_topological_sorts_multigraph_1(self):
    DG = nx.MultiDiGraph([(1, 2), (1, 2), (2, 3), (3, 4), (3, 5), (3, 5), (3, 5)])
    assert sorted(nx.all_topological_sorts(DG)) == sorted([[1, 2, 3, 4, 5], [1, 2, 3, 5, 4]])