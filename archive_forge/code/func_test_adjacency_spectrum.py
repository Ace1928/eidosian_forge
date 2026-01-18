import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_adjacency_spectrum(self):
    """Adjacency eigenvalues"""
    evals = np.array([-np.sqrt(2), 0, np.sqrt(2)])
    e = sorted(nx.adjacency_spectrum(self.P))
    np.testing.assert_almost_equal(e, evals)