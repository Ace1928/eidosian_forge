import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_cartesian_product_size():
    K5 = nx.complete_graph(5)
    P5 = nx.path_graph(5)
    K3 = nx.complete_graph(3)
    G = nx.cartesian_product(P5, K3)
    assert nx.number_of_nodes(G) == 5 * 3
    assert nx.number_of_edges(G) == nx.number_of_edges(P5) * nx.number_of_nodes(K3) + nx.number_of_edges(K3) * nx.number_of_nodes(P5)
    G = nx.cartesian_product(K3, K5)
    assert nx.number_of_nodes(G) == 3 * 5
    assert nx.number_of_edges(G) == nx.number_of_edges(K5) * nx.number_of_nodes(K3) + nx.number_of_edges(K3) * nx.number_of_nodes(K5)