import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_cartesian_product_multigraph():
    G = nx.MultiGraph()
    G.add_edge(1, 2, key=0)
    G.add_edge(1, 2, key=1)
    H = nx.MultiGraph()
    H.add_edge(3, 4, key=0)
    H.add_edge(3, 4, key=1)
    GH = nx.cartesian_product(G, H)
    assert set(GH) == {(1, 3), (2, 3), (2, 4), (1, 4)}
    assert {(frozenset([u, v]), k) for u, v, k in GH.edges(keys=True)} == {(frozenset([u, v]), k) for u, v, k in [((1, 3), (2, 3), 0), ((1, 3), (2, 3), 1), ((1, 3), (1, 4), 0), ((1, 3), (1, 4), 1), ((2, 3), (2, 4), 0), ((2, 3), (2, 4), 1), ((2, 4), (1, 4), 0), ((2, 4), (1, 4), 1)]}