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
def test_n_components():
    rng = np.random.RandomState(42)
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    init = rng.rand(X.shape[1] - 1, 3)
    n_components = X.shape[1]
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) does not match the output dimensionality of the given linear transformation `init` ({init.shape[0]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    n_components = X.shape[1] + 2
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) cannot be greater than the given data dimensionality ({X.shape[1]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    nca = NeighborhoodComponentsAnalysis(n_components=2, init='identity')
    nca.fit(X, y)