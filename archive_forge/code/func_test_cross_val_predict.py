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
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_cross_val_predict(coo_container):
    X, y = load_diabetes(return_X_y=True)
    cv = KFold()
    est = Ridge()
    preds2 = np.zeros_like(y)
    for train, test in cv.split(X, y):
        est.fit(X[train], y[train])
        preds2[test] = est.predict(X[test])
    preds = cross_val_predict(est, X, y, cv=cv)
    assert_array_almost_equal(preds, preds2)
    preds = cross_val_predict(est, X, y)
    assert len(preds) == len(y)
    cv = LeaveOneOut()
    preds = cross_val_predict(est, X, y, cv=cv)
    assert len(preds) == len(y)
    Xsp = X.copy()
    Xsp *= Xsp > np.median(Xsp)
    Xsp = coo_container(Xsp)
    preds = cross_val_predict(est, Xsp, y)
    assert_array_almost_equal(len(preds), len(y))
    preds = cross_val_predict(KMeans(n_init='auto'), X)
    assert len(preds) == len(y)

    class BadCV:

        def split(self, X, y=None, groups=None):
            for i in range(4):
                yield (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7, 8]))
    with pytest.raises(ValueError):
        cross_val_predict(est, X, y, cv=BadCV())
    X, y = load_iris(return_X_y=True)
    warning_message = 'Number of classes in training fold \\(2\\) does not match total number of classes \\(3\\). Results may not be appropriate for your use case.'
    with pytest.warns(RuntimeWarning, match=warning_message):
        cross_val_predict(LogisticRegression(solver='liblinear'), X, y, method='predict_proba', cv=KFold(2))