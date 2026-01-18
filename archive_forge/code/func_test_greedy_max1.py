import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_greedy_max1():
    G = G1()
    B = branchings.greedy_branching(G)
    B_ = build_branching(greedy_subopt_branching_1b)
    assert_equal_branchings(B, B_)