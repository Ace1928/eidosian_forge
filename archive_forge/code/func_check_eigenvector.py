from math import sqrt
import pytest
import networkx as nx
def check_eigenvector(A, l, x):
    nx = np.linalg.norm(x)
    assert nx != pytest.approx(0, abs=1e-07)
    y = A @ x
    ny = np.linalg.norm(y)
    assert x @ y == pytest.approx(nx * ny, abs=1e-07)
    assert ny == pytest.approx(l * nx, abs=1e-07)