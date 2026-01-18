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
def test_hashed_binary_occurrences():
    test_data = ['aaabc', 'abbde']
    vect = HashingVectorizer(alternate_sign=False, analyzer='char', norm=None)
    X = vect.transform(test_data)
    assert np.max(X[0:1].data) == 3
    assert np.max(X[1:2].data) == 2
    assert X.dtype == np.float64
    vect = HashingVectorizer(analyzer='char', alternate_sign=False, binary=True, norm=None)
    X = vect.transform(test_data)
    assert np.max(X.data) == 1
    assert X.dtype == np.float64
    vect = HashingVectorizer(analyzer='char', alternate_sign=False, binary=True, norm=None, dtype=np.float64)
    X = vect.transform(test_data)
    assert X.dtype == np.float64