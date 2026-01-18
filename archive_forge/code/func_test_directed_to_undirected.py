import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_directed_to_undirected(self):
    edges1 = [(0, 1), (1, 2), (2, 0)]
    edges2 = [(0, 1), (1, 2), (0, 2)]
    assert self.edgelists_equal(nx.Graph(nx.DiGraph(edges1)).edges(), edges1)
    assert self.edgelists_equal(nx.Graph(nx.DiGraph(edges2)).edges(), edges1)
    assert self.edgelists_equal(nx.MultiGraph(nx.DiGraph(edges1)).edges(), edges1)
    assert self.edgelists_equal(nx.MultiGraph(nx.DiGraph(edges2)).edges(), edges1)
    assert self.edgelists_equal(nx.MultiGraph(nx.MultiDiGraph(edges1)).edges(), edges1)
    assert self.edgelists_equal(nx.MultiGraph(nx.MultiDiGraph(edges2)).edges(), edges1)
    assert self.edgelists_equal(nx.Graph(nx.MultiDiGraph(edges1)).edges(), edges1)
    assert self.edgelists_equal(nx.Graph(nx.MultiDiGraph(edges2)).edges(), edges1)