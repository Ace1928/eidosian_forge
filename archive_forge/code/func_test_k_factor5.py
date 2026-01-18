import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_k_factor5(self):
    g = gen.complete_graph(6)
    g_kf = reg.k_factor(g, 2)
    for edge in g_kf.edges():
        assert g.has_edge(edge[0], edge[1])
    for _, degree in g_kf.degree():
        assert degree == 2