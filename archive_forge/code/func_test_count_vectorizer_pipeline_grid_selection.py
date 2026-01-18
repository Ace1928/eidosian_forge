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
def test_count_vectorizer_pipeline_grid_selection():
    data = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS
    target = [-1] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)
    train_data, test_data, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=0)
    pipeline = Pipeline([('vect', CountVectorizer()), ('svc', LinearSVC(dual='auto'))])
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'svc__loss': ('hinge', 'squared_hinge')}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, cv=3)
    pred = grid_search.fit(train_data, target_train).predict(test_data)
    assert_array_equal(pred, target_test)
    assert grid_search.best_score_ == 1.0
    best_vectorizer = grid_search.best_estimator_.named_steps['vect']
    assert best_vectorizer.ngram_range == (1, 1)