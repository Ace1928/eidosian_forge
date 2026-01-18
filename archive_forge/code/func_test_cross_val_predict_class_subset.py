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
def test_cross_val_predict_class_subset():
    X = np.arange(200).reshape(100, 2)
    y = np.array([x // 10 for x in range(100)])
    classes = 10
    kfold3 = KFold(n_splits=3)
    kfold4 = KFold(n_splits=4)
    le = LabelEncoder()
    methods = ['decision_function', 'predict_proba', 'predict_log_proba']
    for method in methods:
        est = LogisticRegression(solver='liblinear')
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold3)
        expected_predictions = get_expected_predictions(X, y, kfold3, classes, est, method)
        assert_array_almost_equal(expected_predictions, predictions)
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold4)
        expected_predictions = get_expected_predictions(X, y, kfold4, classes, est, method)
        assert_array_almost_equal(expected_predictions, predictions)
        y = shuffle(np.repeat(range(10), 10), random_state=0)
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold3)
        y = le.fit_transform(y)
        expected_predictions = get_expected_predictions(X, y, kfold3, classes, est, method)
        assert_array_almost_equal(expected_predictions, predictions)