from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_graph_is_tournament():
    for _ in range(10):
        G = random_tournament(5)
        assert is_tournament(G)