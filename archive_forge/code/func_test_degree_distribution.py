from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_degree_distribution(self):
    for n in range(1, 10):
        G = nx.hypercube_graph(n)
        expected_histogram = [0] * n + [2 ** n]
        assert nx.degree_histogram(G) == expected_histogram