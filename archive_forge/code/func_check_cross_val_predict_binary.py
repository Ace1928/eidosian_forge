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
def check_cross_val_predict_binary(est, X, y, method):
    """Helper for tests of cross_val_predict with binary classification"""
    cv = KFold(n_splits=3, shuffle=False)
    if y.ndim == 1:
        exp_shape = (len(X),) if method == 'decision_function' else (len(X), 2)
    else:
        exp_shape = y.shape
    expected_predictions = np.zeros(exp_shape)
    for train, test in cv.split(X, y):
        est = clone(est).fit(X[train], y[train])
        expected_predictions[test] = getattr(est, method)(X[test])
    for tg in [y, y + 1, y - 2, y.astype('str')]:
        assert_allclose(cross_val_predict(est, X, tg, method=method, cv=cv), expected_predictions)