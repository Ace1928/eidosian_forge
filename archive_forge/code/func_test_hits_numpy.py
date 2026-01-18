import pytest
import networkx as nx
from networkx.algorithms.link_analysis.hits_alg import (
def test_hits_numpy(self):
    G = self.G
    h, a = _hits_numpy(G)
    for n in G:
        assert h[n] == pytest.approx(G.h[n], abs=0.0001)
    for n in G:
        assert a[n] == pytest.approx(G.a[n], abs=0.0001)