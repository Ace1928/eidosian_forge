import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_exampleGraph(self):
    c = harmonic_centrality(self.Gb)
    d = {0: 0, 1: 2, 2: 1, 3: 2.5, 4: 1}
    for n in sorted(self.Gb):
        assert c[n] == pytest.approx(d[n], abs=0.001)