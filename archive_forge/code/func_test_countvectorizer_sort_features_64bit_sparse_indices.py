import pickle
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from io import StringIO
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import _IS_WASM, IS_PYPY
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@skip_if_32bit
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_countvectorizer_sort_features_64bit_sparse_indices(csr_container):
    """
    Check that CountVectorizer._sort_features preserves the dtype of its sparse
    feature matrix.

    This test is skipped on 32bit platforms, see:
        https://github.com/scikit-learn/scikit-learn/pull/11295
    for more details.
    """
    X = csr_container((5, 5), dtype=np.int64)
    INDICES_DTYPE = np.int64
    X.indices = X.indices.astype(INDICES_DTYPE)
    X.indptr = X.indptr.astype(INDICES_DTYPE)
    vocabulary = {'scikit-learn': 0, 'is': 1, 'great!': 2}
    Xs = CountVectorizer()._sort_features(X, vocabulary)
    assert INDICES_DTYPE == Xs.indices.dtype