import math
import pytest
import networkx as nx
def test_eigenvector_centrality_unweighted(self):
    G = self.H
    p = nx.eigenvector_centrality(G)
    for a, b in zip(list(p.values()), self.G.evc):
        assert a == pytest.approx(b, abs=0.0001)