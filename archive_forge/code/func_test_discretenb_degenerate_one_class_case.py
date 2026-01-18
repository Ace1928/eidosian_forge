import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('DiscreteNaiveBayes', DISCRETE_NAIVE_BAYES_CLASSES)
@pytest.mark.parametrize('use_partial_fit', [False, True])
@pytest.mark.parametrize('train_on_single_class_y', [False, True])
def test_discretenb_degenerate_one_class_case(DiscreteNaiveBayes, use_partial_fit, train_on_single_class_y):
    X = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    y = [1, 1, 2]
    if train_on_single_class_y:
        X = X[:-1]
        y = y[:-1]
    classes = sorted(list(set(y)))
    num_classes = len(classes)
    clf = DiscreteNaiveBayes()
    if use_partial_fit:
        clf.partial_fit(X, y, classes=classes)
    else:
        clf.fit(X, y)
    assert clf.predict(X[:1]) == y[0]
    attribute_names = ['classes_', 'class_count_', 'class_log_prior_', 'feature_count_', 'feature_log_prob_']
    for attribute_name in attribute_names:
        attribute = getattr(clf, attribute_name, None)
        if attribute is None:
            continue
        if isinstance(attribute, np.ndarray):
            assert attribute.shape[0] == num_classes
        else:
            for element in attribute:
                assert element.shape[0] == num_classes