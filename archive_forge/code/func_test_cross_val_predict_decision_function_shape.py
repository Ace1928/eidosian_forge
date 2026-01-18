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
def test_cross_val_predict_decision_function_shape():
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    preds = cross_val_predict(LogisticRegression(solver='liblinear'), X, y, method='decision_function')
    assert preds.shape == (50,)
    X, y = load_iris(return_X_y=True)
    preds = cross_val_predict(LogisticRegression(solver='liblinear'), X, y, method='decision_function')
    assert preds.shape == (150, 3)
    X = X[:100]
    y = y[:100]
    error_message = 'Only 1 class/es in training fold, but 2 in overall dataset. This is not supported for decision_function with imbalanced folds. To fix this, use a cross-validation technique resulting in properly stratified folds'
    with pytest.raises(ValueError, match=error_message):
        cross_val_predict(RidgeClassifier(), X, y, method='decision_function', cv=KFold(2))
    X, y = load_digits(return_X_y=True)
    est = SVC(kernel='linear', decision_function_shape='ovo')
    preds = cross_val_predict(est, X, y, method='decision_function')
    assert preds.shape == (1797, 45)
    ind = np.argsort(y)
    X, y = (X[ind], y[ind])
    error_message_regexp = 'Output shape \\(599L?, 21L?\\) of decision_function does not match number of classes \\(7\\) in fold. Irregular decision_function .*'
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_val_predict(est, X, y, cv=KFold(n_splits=3), method='decision_function')