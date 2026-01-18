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
@fails_if_pypy
def test_hashing_vectorizer():
    v = HashingVectorizer()
    X = v.transform(ALL_FOOD_DOCS)
    token_nnz = X.nnz
    assert X.shape == (len(ALL_FOOD_DOCS), v.n_features)
    assert X.dtype == v.dtype
    assert np.min(X.data) > -1
    assert np.min(X.data) < 0
    assert np.max(X.data) > 0
    assert np.max(X.data) < 1
    for i in range(X.shape[0]):
        assert_almost_equal(np.linalg.norm(X[0].data, 2), 1.0)
    v = HashingVectorizer(ngram_range=(1, 2), norm='l1')
    X = v.transform(ALL_FOOD_DOCS)
    assert X.shape == (len(ALL_FOOD_DOCS), v.n_features)
    assert X.dtype == v.dtype
    ngrams_nnz = X.nnz
    assert ngrams_nnz > token_nnz
    assert ngrams_nnz < 2 * token_nnz
    assert np.min(X.data) > -1
    assert np.max(X.data) < 1
    for i in range(X.shape[0]):
        assert_almost_equal(np.linalg.norm(X[0].data, 1), 1.0)