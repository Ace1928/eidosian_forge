from math import sqrt
import pytest
import networkx as nx
def test_abbreviation_of_method(self):
    pytest.importorskip('scipy')
    G = nx.path_graph(8)
    A = nx.laplacian_matrix(G)
    sigma = 2 - sqrt(2 + sqrt(2))
    ac = nx.algebraic_connectivity(G, tol=1e-12, method='tracemin')
    assert ac == pytest.approx(sigma, abs=1e-07)
    x = nx.fiedler_vector(G, tol=1e-12, method='tracemin')
    check_eigenvector(A, sigma, x)