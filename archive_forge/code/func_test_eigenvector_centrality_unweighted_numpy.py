import math
import pytest
import networkx as nx
def test_eigenvector_centrality_unweighted_numpy(self):
    G = self.H
    p = nx.eigenvector_centrality_numpy(G)
    for a, b in zip(list(p.values()), self.G.evc):
        assert a == pytest.approx(b, abs=1e-07)