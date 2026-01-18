import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_classification():
    weights = [0.1, 0.25]
    X, y = make_classification(n_samples=100, n_features=20, n_informative=5, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1, hypercube=False, shift=None, scale=None, weights=weights, random_state=0)
    assert weights == [0.1, 0.25]
    assert X.shape == (100, 20), 'X shape mismatch'
    assert y.shape == (100,), 'y shape mismatch'
    assert np.unique(y).shape == (3,), 'Unexpected number of classes'
    assert sum(y == 0) == 10, 'Unexpected number of samples in class #0'
    assert sum(y == 1) == 25, 'Unexpected number of samples in class #1'
    assert sum(y == 2) == 65, 'Unexpected number of samples in class #2'
    X, y = make_classification(n_samples=2000, n_features=31, n_informative=31, n_redundant=0, n_repeated=0, hypercube=True, scale=0.5, random_state=0)
    assert X.shape == (2000, 31), 'X shape mismatch'
    assert y.shape == (2000,), 'y shape mismatch'
    assert np.unique(X.view([('', X.dtype)] * X.shape[1])).view(X.dtype).reshape(-1, X.shape[1]).shape[0] == 2000, 'Unexpected number of unique rows'