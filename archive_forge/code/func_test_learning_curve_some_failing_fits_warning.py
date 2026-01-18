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
def test_learning_curve_some_failing_fits_warning(global_random_seed):
    """Checks for fit failures in `learning_curve` and raises the required warning"""
    X, y = make_classification(n_samples=30, n_classes=3, n_informative=6, shuffle=False, random_state=global_random_seed)
    sorted_idx = np.argsort(y)
    X, y = (X[sorted_idx], y[sorted_idx])
    svc = SVC()
    warning_message = '10 fits failed out of a total of 25'
    with pytest.warns(FitFailedWarning, match=warning_message):
        _, train_score, test_score, *_ = learning_curve(svc, X, y, cv=5, error_score=np.nan)
    for idx in range(2):
        assert np.isnan(train_score[idx]).all()
        assert np.isnan(test_score[idx]).all()
    for idx in range(2, train_score.shape[0]):
        assert not np.isnan(train_score[idx]).any()
        assert not np.isnan(test_score[idx]).any()