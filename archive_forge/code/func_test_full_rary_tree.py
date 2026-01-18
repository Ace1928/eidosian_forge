import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_full_rary_tree(self):
    r = 2
    n = 9
    t = nx.full_rary_tree(r, n)
    assert t.order() == n
    assert nx.is_connected(t)
    dh = nx.degree_histogram(t)
    assert dh[0] == 0
    assert dh[1] == 5
    assert dh[r] == 1
    assert dh[r + 1] == 9 - 5 - 1
    assert len(dh) == r + 2