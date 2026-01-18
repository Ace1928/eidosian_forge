import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_bal_tree(self):
    c = harmonic_centrality(self.T)
    d = {0: 4.0, 1: 4.1666, 2: 4.1666, 3: 2.8333, 4: 2.8333, 5: 2.8333, 6: 2.8333}
    for n in sorted(self.T):
        assert c[n] == pytest.approx(d[n], abs=0.001)