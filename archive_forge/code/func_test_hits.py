import pytest
import networkx as nx
from networkx.algorithms.link_analysis.hits_alg import (
@pytest.mark.parametrize('hits_alg', (nx.hits, _hits_python, _hits_scipy))
def test_hits(self, hits_alg):
    G = self.G
    h, a = hits_alg(G, tol=1e-08)
    for n in G:
        assert h[n] == pytest.approx(G.h[n], abs=0.0001)
    for n in G:
        assert a[n] == pytest.approx(G.a[n], abs=0.0001)
    nstart = {i: 1.0 / 2 for i in G}
    h, a = hits_alg(G, nstart=nstart)
    for n in G:
        assert h[n] == pytest.approx(G.h[n], abs=0.0001)
    for n in G:
        assert a[n] == pytest.approx(G.a[n], abs=0.0001)