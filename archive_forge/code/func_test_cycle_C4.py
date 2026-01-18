import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_cycle_C4(self):
    c = harmonic_centrality(self.C4)
    d = {0: 2.5, 1: 2.5, 2: 2.5, 3: 2.5}
    for n in sorted(self.C4):
        assert c[n] == pytest.approx(d[n], abs=0.001)