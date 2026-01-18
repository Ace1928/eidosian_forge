import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_learning_curve():
    n_samples = 30
    n_splits = 3
    X, y = make_classification(n_samples=n_samples, n_features=1, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=0)
    estimator = MockImprovingEstimator(n_samples * ((n_splits - 1) / n_splits))
    for shuffle_train in [False, True]:
        with warnings.catch_warnings(record=True) as w:
            train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator, X, y, cv=KFold(n_splits=n_splits), train_sizes=np.linspace(0.1, 1.0, 10), shuffle=shuffle_train, return_times=True)
        if len(w) > 0:
            raise RuntimeError('Unexpected warning: %r' % w[0].message)
        assert train_scores.shape == (10, 3)
        assert test_scores.shape == (10, 3)
        assert fit_times.shape == (10, 3)
        assert score_times.shape == (10, 3)
        assert_array_equal(train_sizes, np.linspace(2, 20, 10))
        assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
        assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))
        assert fit_times.dtype == 'float64'
        assert score_times.dtype == 'float64'
        with warnings.catch_warnings(record=True) as w:
            train_sizes2, train_scores2, test_scores2 = learning_curve(estimator, X, y, cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples), train_sizes=np.linspace(0.1, 1.0, 10), shuffle=shuffle_train)
        if len(w) > 0:
            raise RuntimeError('Unexpected warning: %r' % w[0].message)
        assert_array_almost_equal(train_scores2, train_scores)
        assert_array_almost_equal(test_scores2, test_scores)