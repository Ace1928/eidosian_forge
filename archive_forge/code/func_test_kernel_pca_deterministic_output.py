import warnings
import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_blobs, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Perceptron
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _check_psd_eigenvalues
def test_kernel_pca_deterministic_output():
    """Test that Kernel PCA produces deterministic output

    Tests that the same inputs and random state produce the same output.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(10, 10)
    eigen_solver = ('arpack', 'dense')
    for solver in eigen_solver:
        transformed_X = np.zeros((20, 2))
        for i in range(20):
            kpca = KernelPCA(n_components=2, eigen_solver=solver, random_state=rng)
            transformed_X[i, :] = kpca.fit_transform(X)[0]
        assert_allclose(transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2))