from itertools import combinations
import pytest
import networkx as nx
def test_k3(self):
    comm1 = [0, 1, 2, 3, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33]
    comm2 = [0, 4, 5, 6, 10, 16]
    comm3 = [24, 25, 31]
    expected = {frozenset(comm1), frozenset(comm2), frozenset(comm3)}
    self._check_communities(3, expected)