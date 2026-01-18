from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_all_topological_sorts_2(self):
    DG = nx.DiGraph([(1, 3), (2, 1), (2, 4), (4, 3), (4, 5)])
    assert sorted(nx.all_topological_sorts(DG)) == [[2, 1, 4, 3, 5], [2, 1, 4, 5, 3], [2, 4, 1, 3, 5], [2, 4, 1, 5, 3], [2, 4, 5, 1, 3]]