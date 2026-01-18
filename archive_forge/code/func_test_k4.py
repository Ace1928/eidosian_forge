from itertools import combinations
import pytest
import networkx as nx
def test_k4(self):
    expected = {frozenset([0, 1, 2, 3, 7, 13]), frozenset([8, 32, 30, 33]), frozenset([32, 33, 29, 23])}
    self._check_communities(4, expected)