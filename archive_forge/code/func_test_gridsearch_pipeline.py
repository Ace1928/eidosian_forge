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
def test_gridsearch_pipeline():
    """Check that kPCA works as expected in a grid search pipeline

    Test if we can do a grid-search to find parameters to separate
    circles with a perceptron model.
    """
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)
    kpca = KernelPCA(kernel='rbf', n_components=2)
    pipeline = Pipeline([('kernel_pca', kpca), ('Perceptron', Perceptron(max_iter=5))])
    param_grid = dict(kernel_pca__gamma=2.0 ** np.arange(-2, 2))
    grid_search = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
    grid_search.fit(X, y)
    assert grid_search.best_score_ == 1