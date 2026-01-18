import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edmonds2_maxbranch():
    G = G2()
    x = branchings.maximum_branching(G)
    x_ = build_branching(optimal_branching_2a)
    assert_equal_branchings(x, x_)