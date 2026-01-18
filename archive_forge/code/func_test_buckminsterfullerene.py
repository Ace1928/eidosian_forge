from math import sqrt
import pytest
import networkx as nx
@pytest.mark.parametrize(('normalized', 'sigma', 'laplacian_fn'), ((False, 0.2434017461399311, nx.laplacian_matrix), (True, 0.0811339153799775, nx.normalized_laplacian_matrix)))
@pytest.mark.parametrize('method', methods)
def test_buckminsterfullerene(self, normalized, sigma, laplacian_fn, method):
    pytest.importorskip('scipy')
    G = nx.Graph([(1, 10), (1, 41), (1, 59), (2, 12), (2, 42), (2, 60), (3, 6), (3, 43), (3, 57), (4, 8), (4, 44), (4, 58), (5, 13), (5, 56), (5, 57), (6, 10), (6, 31), (7, 14), (7, 56), (7, 58), (8, 12), (8, 32), (9, 23), (9, 53), (9, 59), (10, 15), (11, 24), (11, 53), (11, 60), (12, 16), (13, 14), (13, 25), (14, 26), (15, 27), (15, 49), (16, 28), (16, 50), (17, 18), (17, 19), (17, 54), (18, 20), (18, 55), (19, 23), (19, 41), (20, 24), (20, 42), (21, 31), (21, 33), (21, 57), (22, 32), (22, 34), (22, 58), (23, 24), (25, 35), (25, 43), (26, 36), (26, 44), (27, 51), (27, 59), (28, 52), (28, 60), (29, 33), (29, 34), (29, 56), (30, 51), (30, 52), (30, 53), (31, 47), (32, 48), (33, 45), (34, 46), (35, 36), (35, 37), (36, 38), (37, 39), (37, 49), (38, 40), (38, 50), (39, 40), (39, 51), (40, 52), (41, 47), (42, 48), (43, 49), (44, 50), (45, 46), (45, 54), (46, 55), (47, 54), (48, 55)])
    A = laplacian_fn(G)
    try:
        assert nx.algebraic_connectivity(G, normalized=normalized, tol=1e-12, method=method) == pytest.approx(sigma, abs=1e-07)
        x = nx.fiedler_vector(G, normalized=normalized, tol=1e-12, method=method)
        check_eigenvector(A, sigma, x)
    except nx.NetworkXError as err:
        if err.args not in (('Cholesky solver unavailable.',), ('LU solver unavailable.',)):
            raise