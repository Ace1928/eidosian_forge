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
def test_kernel_conditioning():
    """Check that ``_check_psd_eigenvalues`` is correctly called in kPCA

    Non-regression test for issue #12140 (PR #12145).
    """
    X = [[5, 1], [5 + 1e-08, 1e-08], [5 + 1e-08, 0]]
    kpca = KernelPCA(kernel='linear', n_components=2, fit_inverse_transform=True)
    kpca.fit(X)
    assert kpca.eigenvalues_.min() == 0
    assert np.all(kpca.eigenvalues_ == _check_psd_eigenvalues(kpca.eigenvalues_))