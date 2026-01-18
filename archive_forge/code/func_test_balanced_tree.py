import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_balanced_tree(self):
    for r, h in [(2, 2), (3, 3), (6, 2)]:
        t = nx.balanced_tree(r, h)
        order = t.order()
        assert order == (r ** (h + 1) - 1) / (r - 1)
        assert nx.is_connected(t)
        assert t.size() == order - 1
        dh = nx.degree_histogram(t)
        assert dh[0] == 0
        assert dh[1] == r ** h
        assert dh[r] == 1
        assert dh[r + 1] == order - r ** h - 1
        assert len(dh) == r + 2