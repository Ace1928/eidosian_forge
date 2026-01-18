import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_full_rary_tree_balanced(self):
    t = nx.full_rary_tree(2, 15)
    th = nx.balanced_tree(2, 3)
    assert is_isomorphic(t, th)