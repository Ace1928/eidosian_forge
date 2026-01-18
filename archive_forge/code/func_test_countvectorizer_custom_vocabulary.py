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
def test_countvectorizer_custom_vocabulary():
    vocab = {'pizza': 0, 'beer': 1}
    terms = set(vocab.keys())
    for typ in [dict, list, iter, partial(defaultdict, int)]:
        v = typ(vocab)
        vect = CountVectorizer(vocabulary=v)
        vect.fit(JUNK_FOOD_DOCS)
        if isinstance(v, Mapping):
            assert vect.vocabulary_ == vocab
        else:
            assert set(vect.vocabulary_) == terms
        X = vect.transform(JUNK_FOOD_DOCS)
        assert X.shape[1] == len(terms)
        v = typ(vocab)
        vect = CountVectorizer(vocabulary=v)
        inv = vect.inverse_transform(X)
        assert len(inv) == X.shape[0]