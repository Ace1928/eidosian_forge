import pytest
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.degree_seq import havel_hakimi_graph
def test_adjacency_matrix(self):
    """Conversion to adjacency matrix"""
    np.testing.assert_equal(nx.adjacency_matrix(self.G).todense(), self.A)
    np.testing.assert_equal(nx.adjacency_matrix(self.MG).todense(), self.A)
    np.testing.assert_equal(nx.adjacency_matrix(self.MG2).todense(), self.MG2A)
    np.testing.assert_equal(nx.adjacency_matrix(self.G, nodelist=[0, 1]).todense(), self.A[:2, :2])
    np.testing.assert_equal(nx.adjacency_matrix(self.WG).todense(), self.WA)
    np.testing.assert_equal(nx.adjacency_matrix(self.WG, weight=None).todense(), self.A)
    np.testing.assert_equal(nx.adjacency_matrix(self.MG2, weight=None).todense(), self.MG2A)
    np.testing.assert_equal(nx.adjacency_matrix(self.WG, weight='other').todense(), 0.6 * self.WA)
    np.testing.assert_equal(nx.adjacency_matrix(self.no_edges_G, nodelist=[1, 3]).todense(), self.no_edges_A)