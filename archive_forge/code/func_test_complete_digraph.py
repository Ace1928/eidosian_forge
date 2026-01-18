import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_complete_digraph(self):
    for m in [0, 1, 3, 5]:
        g = nx.complete_graph(m, create_using=nx.DiGraph)
        assert nx.number_of_nodes(g) == m
        assert nx.number_of_edges(g) == m * (m - 1)
    g = nx.complete_graph('abc', create_using=nx.DiGraph)
    assert len(g) == 3
    assert g.size() == 6
    assert g.is_directed()