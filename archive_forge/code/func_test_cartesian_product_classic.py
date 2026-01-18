import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_cartesian_product_classic():
    P2 = nx.path_graph(2)
    P3 = nx.path_graph(3)
    G = nx.cartesian_product(P2, P2)
    G = nx.cartesian_product(P2, G)
    assert nx.is_isomorphic(G, nx.cubical_graph())
    G = nx.cartesian_product(P3, P3)
    assert nx.is_isomorphic(G, nx.grid_2d_graph(3, 3))