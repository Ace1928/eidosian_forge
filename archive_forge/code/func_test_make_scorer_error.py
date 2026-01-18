import numbers
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.metrics import (
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics._scorer import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
from sklearn.utils.metadata_routing import MetadataRouter
@pytest.mark.filterwarnings('ignore:.*needs_proba.*:FutureWarning')
@pytest.mark.parametrize('params, err_type, err_msg', [({'response_method': 'predict_proba', 'needs_proba': True}, ValueError, 'You cannot set both `response_method`'), ({'response_method': 'predict_proba', 'needs_threshold': True}, ValueError, 'You cannot set both `response_method`'), ({'needs_proba': True, 'needs_threshold': True}, ValueError, 'You cannot set both `needs_proba` and `needs_threshold`')])
def test_make_scorer_error(params, err_type, err_msg):
    """Check that `make_scorer` raises errors if the parameter used."""
    with pytest.raises(err_type, match=err_msg):
        make_scorer(lambda y_true, y_pred: 1, **params)