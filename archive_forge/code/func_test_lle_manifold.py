from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import manifold, neighbors
from sklearn.datasets import make_blobs
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from sklearn.utils._testing import (
@pytest.mark.parametrize('method', ['standard', 'hessian', 'modified', 'ltsa'])
@pytest.mark.parametrize('solver', eigen_solvers)
def test_lle_manifold(global_dtype, method, solver):
    rng = np.random.RandomState(0)
    X = np.array(list(product(np.arange(18), repeat=2)))
    X = np.c_[X, X[:, 0] ** 2 / 18]
    X = X + 1e-10 * rng.uniform(size=X.shape)
    X = X.astype(global_dtype, copy=False)
    n_components = 2
    clf = manifold.LocallyLinearEmbedding(n_neighbors=6, n_components=n_components, method=method, random_state=0)
    tol = 1.5 if method == 'standard' else 3
    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()
    reconstruction_error = linalg.norm(np.dot(N, X) - X)
    assert reconstruction_error < tol
    clf.set_params(eigen_solver=solver)
    clf.fit(X)
    assert clf.embedding_.shape[1] == n_components
    reconstruction_error = linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, 'fro') ** 2
    details = 'solver: %s, method: %s' % (solver, method)
    assert reconstruction_error < tol, details
    assert np.abs(clf.reconstruction_error_ - reconstruction_error) < tol * reconstruction_error, details