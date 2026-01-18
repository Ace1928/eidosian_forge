import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_laplacian_spectrum(self):
    """Laplacian eigenvalues"""
    evals = np.array([0, 0, 1, 3, 4])
    e = sorted(nx.laplacian_spectrum(self.G))
    np.testing.assert_almost_equal(e, evals)
    e = sorted(nx.laplacian_spectrum(self.WG, weight=None))
    np.testing.assert_almost_equal(e, evals)
    e = sorted(nx.laplacian_spectrum(self.WG))
    np.testing.assert_almost_equal(e, 0.5 * evals)
    e = sorted(nx.laplacian_spectrum(self.WG, weight='other'))
    np.testing.assert_almost_equal(e, 0.3 * evals)