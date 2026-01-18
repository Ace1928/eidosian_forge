import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_lexicographic_product_size():
    K5 = nx.complete_graph(5)
    P5 = nx.path_graph(5)
    K3 = nx.complete_graph(3)
    G = nx.lexicographic_product(P5, K3)
    assert nx.number_of_nodes(G) == 5 * 3
    G = nx.lexicographic_product(K3, K5)
    assert nx.number_of_nodes(G) == 3 * 5