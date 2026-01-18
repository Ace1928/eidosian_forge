from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_lexicographical_topological_sort(self):
    G = nx.DiGraph([(1, 2), (2, 3), (1, 4), (1, 5), (2, 6)])
    assert list(nx.lexicographical_topological_sort(G)) == [1, 2, 3, 4, 5, 6]
    assert list(nx.lexicographical_topological_sort(G, key=lambda x: x)) == [1, 2, 3, 4, 5, 6]
    assert list(nx.lexicographical_topological_sort(G, key=lambda x: -x)) == [1, 5, 4, 2, 6, 3]