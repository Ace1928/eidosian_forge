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
def test_cross_validate_return_indices(global_random_seed):
    """Check the behaviour of `return_indices` in `cross_validate`."""
    X, y = load_iris(return_X_y=True)
    X = scale(X)
    estimator = LogisticRegression()
    cv = KFold(n_splits=3, shuffle=True, random_state=global_random_seed)
    cv_results = cross_validate(estimator, X, y, cv=cv, n_jobs=2, return_indices=False)
    assert 'indices' not in cv_results
    cv_results = cross_validate(estimator, X, y, cv=cv, n_jobs=2, return_indices=True)
    assert 'indices' in cv_results
    train_indices = cv_results['indices']['train']
    test_indices = cv_results['indices']['test']
    assert len(train_indices) == cv.n_splits
    assert len(test_indices) == cv.n_splits
    assert_array_equal([indices.size for indices in train_indices], 100)
    assert_array_equal([indices.size for indices in test_indices], 50)
    for split_idx, (expected_train_idx, expected_test_idx) in enumerate(cv.split(X, y)):
        assert_array_equal(train_indices[split_idx], expected_train_idx)
        assert_array_equal(test_indices[split_idx], expected_test_idx)