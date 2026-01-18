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
@pytest.mark.parametrize('solver', ['auto', 'dense', 'arpack', 'randomized'])
@pytest.mark.parametrize('n_features', [4, 10])
def test_kernel_pca_linear_kernel(solver, n_features):
    """Test that kPCA with linear kernel is equivalent to PCA for all solvers.

    KernelPCA with linear kernel should produce the same output as PCA.
    """
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, n_features))
    X_pred = rng.random_sample((2, n_features))
    n_comps = 3 if solver == 'arpack' else 4
    assert_array_almost_equal(np.abs(KernelPCA(n_comps, eigen_solver=solver).fit(X_fit).transform(X_pred)), np.abs(PCA(n_comps, svd_solver=solver if solver != 'dense' else 'full').fit(X_fit).transform(X_pred)))