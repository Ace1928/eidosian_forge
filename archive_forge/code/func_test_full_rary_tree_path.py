import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_full_rary_tree_path(self):
    t = nx.full_rary_tree(1, 10)
    assert is_isomorphic(t, nx.path_graph(10))