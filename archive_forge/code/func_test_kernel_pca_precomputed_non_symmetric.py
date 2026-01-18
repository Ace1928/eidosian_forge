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
def test_kernel_pca_precomputed_non_symmetric(solver):
    """Check that the kernel centerer works.

    Tests that a non symmetric precomputed kernel is actually accepted
    because the kernel centerer does its job correctly.
    """
    K = [[1, 2], [3, 40]]
    kpca = KernelPCA(kernel='precomputed', eigen_solver=solver, n_components=1, random_state=0)
    kpca.fit(K)
    Kc = [[9, -9], [-9, 9]]
    kpca_c = KernelPCA(kernel='precomputed', eigen_solver=solver, n_components=1, random_state=0)
    kpca_c.fit(Kc)
    assert_array_equal(kpca.eigenvectors_, kpca_c.eigenvectors_)
    assert_array_equal(kpca.eigenvalues_, kpca_c.eigenvalues_)