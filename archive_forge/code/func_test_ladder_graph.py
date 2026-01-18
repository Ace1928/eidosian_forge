import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_ladder_graph(self):
    for i, G in [(0, nx.empty_graph(0)), (1, nx.path_graph(2)), (2, nx.hypercube_graph(2)), (10, nx.grid_graph([2, 10]))]:
        assert is_isomorphic(nx.ladder_graph(i), G)
    pytest.raises(nx.NetworkXError, nx.ladder_graph, 2, create_using=nx.DiGraph)
    g = nx.ladder_graph(2)
    mg = nx.ladder_graph(2, create_using=nx.MultiGraph)
    assert edges_equal(mg.edges(), g.edges())