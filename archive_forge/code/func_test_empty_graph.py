import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_empty_graph(self):
    G = nx.empty_graph()
    assert nx.number_of_nodes(G) == 0
    G = nx.empty_graph(42)
    assert nx.number_of_nodes(G) == 42
    assert nx.number_of_edges(G) == 0
    G = nx.empty_graph('abc')
    assert len(G) == 3
    assert G.size() == 0
    G = nx.empty_graph(42, create_using=nx.DiGraph(name='duh'))
    assert nx.number_of_nodes(G) == 42
    assert nx.number_of_edges(G) == 0
    assert isinstance(G, nx.DiGraph)
    G = nx.empty_graph(42, create_using=nx.MultiGraph(name='duh'))
    assert nx.number_of_nodes(G) == 42
    assert nx.number_of_edges(G) == 0
    assert isinstance(G, nx.MultiGraph)
    pete = nx.petersen_graph()
    G = nx.empty_graph(42, create_using=pete)
    assert nx.number_of_nodes(G) == 42
    assert nx.number_of_edges(G) == 0
    assert isinstance(G, nx.Graph)