import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_multi_target_regression_partial_fit():
    X, y = datasets.make_regression(n_targets=3, random_state=0)
    X_train, y_train = (X[:50], y[:50])
    X_test, y_test = (X[50:], y[50:])
    references = np.zeros_like(y_test)
    half_index = 25
    for n in range(3):
        sgr = SGDRegressor(random_state=0, max_iter=5)
        sgr.partial_fit(X_train[:half_index], y_train[:half_index, n])
        sgr.partial_fit(X_train[half_index:], y_train[half_index:, n])
        references[:, n] = sgr.predict(X_test)
    sgr = MultiOutputRegressor(SGDRegressor(random_state=0, max_iter=5))
    sgr.partial_fit(X_train[:half_index], y_train[:half_index])
    sgr.partial_fit(X_train[half_index:], y_train[half_index:])
    y_pred = sgr.predict(X_test)
    assert_almost_equal(references, y_pred)
    assert not hasattr(MultiOutputRegressor(Lasso), 'partial_fit')