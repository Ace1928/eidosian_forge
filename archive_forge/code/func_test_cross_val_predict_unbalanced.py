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
def test_cross_val_predict_unbalanced():
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=1)
    y[0] = 2
    clf = LogisticRegression(random_state=1, solver='liblinear')
    cv = StratifiedKFold(n_splits=2)
    train, test = list(cv.split(X, y))
    yhat_proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
    assert y[test[0]][0] == 2
    assert np.all(yhat_proba[test[0]][:, 2] == 0)
    assert np.all(yhat_proba[test[0]][:, 0:1] > 0)
    assert np.all(yhat_proba[test[1]] > 0)
    assert_array_almost_equal(yhat_proba.sum(axis=1), np.ones(y.shape), decimal=12)