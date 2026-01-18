from itertools import combinations
import pytest
import networkx as nx
def test_k2(self):
    expected = {frozenset(self.G)}
    self._check_communities(2, expected)