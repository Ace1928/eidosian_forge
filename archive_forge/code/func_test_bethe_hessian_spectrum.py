import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_bethe_hessian_spectrum(self):
    """Bethe Hessian eigenvalues"""
    evals = np.array([0.5 * (9 - np.sqrt(33)), 4, 0.5 * (9 + np.sqrt(33))])
    e = sorted(nx.bethe_hessian_spectrum(self.P, r=2))
    np.testing.assert_almost_equal(e, evals)
    e1 = sorted(nx.bethe_hessian_spectrum(self.P, r=1))
    e2 = sorted(nx.laplacian_spectrum(self.P))
    np.testing.assert_almost_equal(e1, e2)