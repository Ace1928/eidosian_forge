import math
import pytest
import networkx as nx
def test_P3_unweighted(self):
    """Eigenvector centrality: P3"""
    G = nx.path_graph(3)
    b_answer = {0: 0.5, 1: 0.7071, 2: 0.5}
    b = nx.eigenvector_centrality_numpy(G, weight=None)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=0.0001)