import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_circular_ladder_graph(self):
    G = nx.circular_ladder_graph(5)
    pytest.raises(nx.NetworkXError, nx.circular_ladder_graph, 5, create_using=nx.DiGraph)
    mG = nx.circular_ladder_graph(5, create_using=nx.MultiGraph)
    assert edges_equal(mG.edges(), G.edges())