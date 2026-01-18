import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edmonds1_maxarbor():
    G = G1()
    x = branchings.maximum_spanning_arborescence(G)
    x_ = build_branching(optimal_arborescence_1)
    assert_equal_branchings(x, x_)