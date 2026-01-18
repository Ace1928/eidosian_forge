import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_assortativity_node_kwargs(self):
    G = nx.Graph()
    G.add_nodes_from([0, 1], size=2)
    G.add_nodes_from([2, 3], size=3)
    G.add_edges_from([(0, 1), (2, 3)])
    r = nx.numeric_assortativity_coefficient(G, 'size', nodes=[0, 3])
    np.testing.assert_almost_equal(r, 1.0, decimal=4)