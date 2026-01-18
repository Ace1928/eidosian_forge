import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edmonds1_minimal_branching():
    G = nx.from_numpy_array(G_big_array, create_using=nx.DiGraph)
    B = branchings.minimal_branching(G)
    edges = [(3, 0, 5), (0, 2, 12), (0, 4, 12), (2, 5, 12), (4, 7, 12), (5, 8, 12), (5, 6, 14), (2, 1, 17)]
    B_ = build_branching(edges, double=True)
    assert_equal_branchings(B, B_)