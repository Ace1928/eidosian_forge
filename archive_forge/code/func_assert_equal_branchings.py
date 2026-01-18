import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def assert_equal_branchings(G1, G2, attr='weight', default=1):
    edges1 = list(G1.edges(data=True))
    edges2 = list(G2.edges(data=True))
    assert len(edges1) == len(edges2)
    e1 = sorted_edges(G1, attr, default)
    e2 = sorted_edges(G2, attr, default)
    for a, b in zip(e1, e2):
        assert a[:2] == b[:2]
        np.testing.assert_almost_equal(a[2], b[2])