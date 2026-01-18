import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edmonds2_minarbor():
    G = G1()
    x = branchings.minimum_spanning_arborescence(G)
    edges = [(3, 0, 5), (0, 2, 12), (0, 4, 12), (2, 5, 12), (4, 7, 12), (5, 8, 12), (5, 6, 14), (2, 1, 17)]
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)