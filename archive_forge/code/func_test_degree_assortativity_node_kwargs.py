import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_assortativity_node_kwargs(self):
    G = nx.Graph()
    edges = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (5, 9), (9, 0)]
    G.add_edges_from(edges)
    r = nx.degree_assortativity_coefficient(G, nodes=[1, 2, 4])
    np.testing.assert_almost_equal(r, -1.0, decimal=4)