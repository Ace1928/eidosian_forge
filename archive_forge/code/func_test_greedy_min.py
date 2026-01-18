import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_greedy_min():
    G = G1()
    B = branchings.greedy_branching(G, kind='min')
    edges = [(1, 0, 4), (0, 2, 12), (0, 4, 12), (2, 5, 12), (4, 7, 12), (5, 8, 12), (5, 6, 14), (7, 3, 19)]
    B_ = build_branching(edges)
    assert_equal_branchings(B, B_)