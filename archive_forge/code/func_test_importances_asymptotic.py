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
def test_importances_asymptotic():

    def binomial(k, n):
        return 0 if k < 0 or k > n else comb(int(n), int(k), exact=True)

    def entropy(samples):
        n_samples = len(samples)
        entropy = 0.0
        for count in np.bincount(samples):
            p = 1.0 * count / n_samples
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def mdi_importance(X_m, X, y):
        n_samples, n_features = X.shape
        features = list(range(n_features))
        features.pop(X_m)
        values = [np.unique(X[:, i]) for i in range(n_features)]
        imp = 0.0
        for k in range(n_features):
            coef = 1.0 / (binomial(k, n_features) * (n_features - k))
            for B in combinations(features, k):
                for b in product(*[values[B[j]] for j in range(k)]):
                    mask_b = np.ones(n_samples, dtype=bool)
                    for j in range(k):
                        mask_b &= X[:, B[j]] == b[j]
                    X_, y_ = (X[mask_b, :], y[mask_b])
                    n_samples_b = len(X_)
                    if n_samples_b > 0:
                        children = []
                        for xi in values[X_m]:
                            mask_xi = X_[:, X_m] == xi
                            children.append(y_[mask_xi])
                        imp += coef * (1.0 * n_samples_b / n_samples) * (entropy(y_) - sum([entropy(c) * len(c) / n_samples_b for c in children]))
        return imp
    data = np.array([[0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1, 2], [1, 0, 1, 1, 0, 1, 1, 3], [0, 1, 1, 1, 0, 1, 0, 4], [1, 1, 0, 1, 0, 1, 1, 5], [1, 1, 0, 1, 1, 1, 1, 6], [1, 0, 1, 0, 0, 1, 0, 7], [1, 1, 1, 1, 1, 1, 1, 8], [1, 1, 1, 1, 0, 1, 1, 9], [1, 1, 1, 0, 1, 1, 1, 0]])
    X, y = (np.array(data[:, :7], dtype=bool), data[:, 7])
    n_features = X.shape[1]
    true_importances = np.zeros(n_features)
    for i in range(n_features):
        true_importances[i] = mdi_importance(i, X, y)
    clf = ExtraTreesClassifier(n_estimators=500, max_features=1, criterion='log_loss', random_state=0).fit(X, y)
    importances = sum((tree.tree_.compute_feature_importances(normalize=False) for tree in clf.estimators_)) / clf.n_estimators
    assert_almost_equal(entropy(y), sum(importances))
    assert np.abs(true_importances - importances).mean() < 0.01