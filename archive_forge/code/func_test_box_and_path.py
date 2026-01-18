import pytest
import networkx as nx
def test_box_and_path(self):
    """Edge betweenness subset centrality: box and path"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
    b_answer = dict.fromkeys(G.edges(), 0)
    b_answer[0, 1] = b_answer[0, 2] = 0.5
    b_answer[1, 3] = b_answer[2, 3] = 0.5
    b_answer[3, 4] = 0.5
    b = nx.edge_betweenness_centrality_subset(G, sources=[0], targets=[3, 4], weight=None)
    for n in sorted(G.edges()):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)