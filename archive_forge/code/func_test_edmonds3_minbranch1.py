import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edmonds3_minbranch1():
    G = G1()
    x = branchings.minimum_branching(G)
    edges = []
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)