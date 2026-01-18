import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_lexicographic_product_random():
    G = nx.erdos_renyi_graph(10, 2 / 10.0)
    H = nx.erdos_renyi_graph(10, 2 / 10.0)
    GH = nx.lexicographic_product(G, H)
    for u_G, u_H in GH.nodes():
        for v_G, v_H in GH.nodes():
            if G.has_edge(u_G, v_G) or (u_G == v_G and H.has_edge(u_H, v_H)):
                assert GH.has_edge((u_G, u_H), (v_G, v_H))
            else:
                assert not GH.has_edge((u_G, u_H), (v_G, v_H))