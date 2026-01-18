import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_binomial_tree(self):
    graphs = (None, nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    for create_using in graphs:
        for n in range(4):
            b = nx.binomial_tree(n, create_using)
            assert nx.number_of_nodes(b) == 2 ** n
            assert nx.number_of_edges(b) == 2 ** n - 1