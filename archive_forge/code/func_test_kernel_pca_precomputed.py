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
def test_kernel_pca_precomputed():
    """Test that kPCA works with a precomputed kernel, for all solvers"""
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    X_pred = rng.random_sample((2, 4))
    for eigen_solver in ('dense', 'arpack', 'randomized'):
        X_kpca = KernelPCA(4, eigen_solver=eigen_solver, random_state=0).fit(X_fit).transform(X_pred)
        X_kpca2 = KernelPCA(4, eigen_solver=eigen_solver, kernel='precomputed', random_state=0).fit(np.dot(X_fit, X_fit.T)).transform(np.dot(X_pred, X_fit.T))
        X_kpca_train = KernelPCA(4, eigen_solver=eigen_solver, kernel='precomputed', random_state=0).fit_transform(np.dot(X_fit, X_fit.T))
        X_kpca_train2 = KernelPCA(4, eigen_solver=eigen_solver, kernel='precomputed', random_state=0).fit(np.dot(X_fit, X_fit.T)).transform(np.dot(X_fit, X_fit.T))
        assert_array_almost_equal(np.abs(X_kpca), np.abs(X_kpca2))
        assert_array_almost_equal(np.abs(X_kpca_train), np.abs(X_kpca_train2))