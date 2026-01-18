import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_fast_versions_properties_threshold_graphs(self):
    cs = 'ddiiddid'
    G = nxt.threshold_graph(cs)
    assert nxt.density('ddiiddid') == nx.density(G)
    assert sorted(nxt.degree_sequence(cs)) == sorted((d for n, d in G.degree()))
    ts = nxt.triangle_sequence(cs)
    assert ts == list(nx.triangles(G).values())
    assert sum(ts) // 3 == nxt.triangles(cs)
    c1 = nxt.cluster_sequence(cs)
    c2 = list(nx.clustering(G).values())
    assert sum((abs(c - d) for c, d in zip(c1, c2))) == pytest.approx(0, abs=1e-07)
    b1 = nx.betweenness_centrality(G).values()
    b2 = nxt.betweenness_sequence(cs)
    assert sum((abs(c - d) for c, d in zip(b1, b2))) < 1e-07
    assert nxt.eigenvalues(cs) == [0, 1, 3, 3, 5, 7, 7, 8]
    assert abs(nxt.degree_correlation(cs) + 0.593038821954) < 1e-12
    assert nxt.degree_correlation('diiiddi') == -0.8
    assert nxt.degree_correlation('did') == -1.0
    assert nxt.degree_correlation('ddd') == 1.0
    assert nxt.eigenvalues('dddiii') == [0, 0, 0, 0, 3, 3]
    assert nxt.eigenvalues('dddiiid') == [0, 1, 1, 1, 4, 4, 7]