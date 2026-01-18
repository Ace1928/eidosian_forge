import pytest
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.degree_seq import havel_hakimi_graph
def test_incidence_matrix(self):
    """Conversion to incidence matrix"""
    I = nx.incidence_matrix(self.G, nodelist=sorted(self.G), edgelist=sorted(self.G.edges()), oriented=True, dtype=int).todense()
    np.testing.assert_equal(I, self.OI)
    I = nx.incidence_matrix(self.G, nodelist=sorted(self.G), edgelist=sorted(self.G.edges()), oriented=False, dtype=int).todense()
    np.testing.assert_equal(I, np.abs(self.OI))
    I = nx.incidence_matrix(self.MG, nodelist=sorted(self.MG), edgelist=sorted(self.MG.edges()), oriented=True, dtype=int).todense()
    np.testing.assert_equal(I, self.OI)
    I = nx.incidence_matrix(self.MG, nodelist=sorted(self.MG), edgelist=sorted(self.MG.edges()), oriented=False, dtype=int).todense()
    np.testing.assert_equal(I, np.abs(self.OI))
    I = nx.incidence_matrix(self.MG2, nodelist=sorted(self.MG2), edgelist=sorted(self.MG2.edges()), oriented=True, dtype=int).todense()
    np.testing.assert_equal(I, self.MGOI)
    I = nx.incidence_matrix(self.MG2, nodelist=sorted(self.MG), edgelist=sorted(self.MG2.edges()), oriented=False, dtype=int).todense()
    np.testing.assert_equal(I, np.abs(self.MGOI))
    I = nx.incidence_matrix(self.G, dtype=np.uint8)
    assert I.dtype == np.uint8