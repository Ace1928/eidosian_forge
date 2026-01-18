import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_cycle_c4_directed(self):
    c = harmonic_centrality(self.C4_directed, nbunch=[0, 1], sources=[1, 2])
    d = {0: 0.833, 1: 0.333}
    for n in [0, 1]:
        assert c[n] == pytest.approx(d[n], abs=0.001)