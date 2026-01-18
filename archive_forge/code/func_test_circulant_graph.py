import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_circulant_graph(self):
    Ci6_1 = nx.circulant_graph(6, [1])
    C6 = nx.cycle_graph(6)
    assert edges_equal(Ci6_1.edges(), C6.edges())
    Ci7 = nx.circulant_graph(7, [1, 2, 3])
    K7 = nx.complete_graph(7)
    assert edges_equal(Ci7.edges(), K7.edges())
    Ci6_1_3 = nx.circulant_graph(6, [1, 3])
    K3_3 = nx.complete_bipartite_graph(3, 3)
    assert is_isomorphic(Ci6_1_3, K3_3)