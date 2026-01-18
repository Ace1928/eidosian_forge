import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
def test_init_transformation():
    rng = np.random.RandomState(42)
    X, y = make_blobs(n_samples=30, centers=6, n_features=5, random_state=0)
    nca = NeighborhoodComponentsAnalysis(init='identity')
    nca.fit(X, y)
    nca_random = NeighborhoodComponentsAnalysis(init='random')
    nca_random.fit(X, y)
    nca_auto = NeighborhoodComponentsAnalysis(init='auto')
    nca_auto.fit(X, y)
    nca_pca = NeighborhoodComponentsAnalysis(init='pca')
    nca_pca.fit(X, y)
    nca_lda = NeighborhoodComponentsAnalysis(init='lda')
    nca_lda.fit(X, y)
    init = rng.rand(X.shape[1], X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    nca.fit(X, y)
    init = rng.rand(X.shape[1], X.shape[1] + 1)
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = f'The input dimensionality ({init.shape[1]}) of the given linear transformation `init` must match the dimensionality of the given inputs `X` ({X.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    init = rng.rand(X.shape[1] + 1, X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = f'The output dimensionality ({init.shape[0]}) of the given linear transformation `init` cannot be greater than its input dimensionality ({init.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    init = rng.rand(X.shape[1], X.shape[1])
    n_components = X.shape[1] - 2
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) does not match the output dimensionality of the given linear transformation `init` ({init.shape[0]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)