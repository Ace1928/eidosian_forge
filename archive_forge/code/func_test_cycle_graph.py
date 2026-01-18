import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_cycle_graph(self):
    G = nx.cycle_graph(4)
    assert edges_equal(G.edges(), [(0, 1), (0, 3), (1, 2), (2, 3)])
    mG = nx.cycle_graph(4, create_using=nx.MultiGraph)
    assert edges_equal(mG.edges(), [(0, 1), (0, 3), (1, 2), (2, 3)])
    G = nx.cycle_graph(4, create_using=nx.DiGraph)
    assert not G.has_edge(2, 1)
    assert G.has_edge(1, 2)
    assert G.is_directed()
    G = nx.cycle_graph('abc')
    assert len(G) == 3
    assert G.size() == 3
    G = nx.cycle_graph('abcb')
    assert len(G) == 3
    assert G.size() == 2
    g = nx.cycle_graph('abc', nx.DiGraph)
    assert len(g) == 3
    assert g.size() == 3
    assert g.is_directed()
    g = nx.cycle_graph('abcb', nx.DiGraph)
    assert len(g) == 3
    assert g.size() == 4