import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_greedy_max3():
    G = G1()
    B = branchings.greedy_branching(G, attr=None)
    edges = [(2, 1, 1), (3, 0, 1), (3, 4, 1), (5, 8, 1), (6, 2, 1), (7, 3, 1), (7, 6, 1), (8, 7, 1)]
    B_ = build_branching(edges)
    assert_equal_branchings(B, B_, default=1)