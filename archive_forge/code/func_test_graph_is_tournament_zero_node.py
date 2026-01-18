from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_graph_is_tournament_zero_node():
    G = random_tournament(0)
    assert is_tournament(G)