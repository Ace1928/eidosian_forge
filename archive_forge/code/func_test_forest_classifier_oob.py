import itertools
import math
import pickle
from collections import defaultdict
from functools import partial
from itertools import combinations, product
from typing import Any, Dict
from unittest.mock import patch
import joblib
import numpy as np
import pytest
from scipy.special import comb
import sklearn
from sklearn import clone, datasets
from sklearn.datasets import make_classification, make_hastie_10_2
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._forest import (
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree._classes import SPARSE_SPLITTERS
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('ForestClassifier', FOREST_CLASSIFIERS.values())
@pytest.mark.parametrize('X_type', ['array', 'sparse_csr', 'sparse_csc'])
@pytest.mark.parametrize('X, y, lower_bound_accuracy', [(*datasets.make_classification(n_samples=300, n_classes=2, random_state=0), 0.9), (*datasets.make_classification(n_samples=1000, n_classes=3, n_informative=6, random_state=0), 0.65), (iris.data, iris.target * 2 + 1, 0.65), (*datasets.make_multilabel_classification(n_samples=300, random_state=0), 0.18)])
@pytest.mark.parametrize('oob_score', [True, partial(f1_score, average='micro')])
def test_forest_classifier_oob(ForestClassifier, X, y, X_type, lower_bound_accuracy, oob_score):
    """Check that OOB score is close to score on a test set."""
    X = _convert_container(X, constructor_name=X_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    classifier = ForestClassifier(n_estimators=40, bootstrap=True, oob_score=oob_score, random_state=0)
    assert not hasattr(classifier, 'oob_score_')
    assert not hasattr(classifier, 'oob_decision_function_')
    classifier.fit(X_train, y_train)
    if callable(oob_score):
        test_score = oob_score(y_test, classifier.predict(X_test))
    else:
        test_score = classifier.score(X_test, y_test)
        assert classifier.oob_score_ >= lower_bound_accuracy
    assert abs(test_score - classifier.oob_score_) <= 0.1
    assert hasattr(classifier, 'oob_score_')
    assert not hasattr(classifier, 'oob_prediction_')
    assert hasattr(classifier, 'oob_decision_function_')
    if y.ndim == 1:
        expected_shape = (X_train.shape[0], len(set(y)))
    else:
        expected_shape = (X_train.shape[0], len(set(y[:, 0])), y.shape[1])
    assert classifier.oob_decision_function_.shape == expected_shape