import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_dorogovtsev_goltsev_mendes_graph(self):
    G = nx.dorogovtsev_goltsev_mendes_graph(0)
    assert edges_equal(G.edges(), [(0, 1)])
    assert nodes_equal(list(G), [0, 1])
    G = nx.dorogovtsev_goltsev_mendes_graph(1)
    assert edges_equal(G.edges(), [(0, 1), (0, 2), (1, 2)])
    assert nx.average_clustering(G) == 1.0
    assert sorted(nx.triangles(G).values()) == [1, 1, 1]
    G = nx.dorogovtsev_goltsev_mendes_graph(10)
    assert nx.number_of_nodes(G) == 29526
    assert nx.number_of_edges(G) == 59049
    assert G.degree(0) == 1024
    assert G.degree(1) == 1024
    assert G.degree(2) == 1024
    pytest.raises(nx.NetworkXError, nx.dorogovtsev_goltsev_mendes_graph, 7, create_using=nx.DiGraph)
    pytest.raises(nx.NetworkXError, nx.dorogovtsev_goltsev_mendes_graph, 7, create_using=nx.MultiGraph)