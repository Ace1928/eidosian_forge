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
def test_countvectorizer_stop_words():
    cv = CountVectorizer()
    cv.set_params(stop_words='english')
    assert cv.get_stop_words() == ENGLISH_STOP_WORDS
    cv.set_params(stop_words='_bad_str_stop_')
    with pytest.raises(ValueError):
        cv.get_stop_words()
    cv.set_params(stop_words='_bad_unicode_stop_')
    with pytest.raises(ValueError):
        cv.get_stop_words()
    stoplist = ['some', 'other', 'words']
    cv.set_params(stop_words=stoplist)
    assert cv.get_stop_words() == set(stoplist)