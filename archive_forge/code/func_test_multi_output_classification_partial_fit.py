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
def test_multi_output_classification_partial_fit():
    sgd_linear_clf = SGDClassifier(loss='log_loss', random_state=1, max_iter=5)
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)
    half_index = X.shape[0] // 2
    multi_target_linear.partial_fit(X[:half_index], y[:half_index], classes=classes)
    first_predictions = multi_target_linear.predict(X)
    assert (n_samples, n_outputs) == first_predictions.shape
    multi_target_linear.partial_fit(X[half_index:], y[half_index:])
    second_predictions = multi_target_linear.predict(X)
    assert (n_samples, n_outputs) == second_predictions.shape
    for i in range(3):
        sgd_linear_clf = clone(sgd_linear_clf)
        sgd_linear_clf.partial_fit(X[:half_index], y[:half_index, i], classes=classes[i])
        assert_array_equal(sgd_linear_clf.predict(X), first_predictions[:, i])
        sgd_linear_clf.partial_fit(X[half_index:], y[half_index:, i])
        assert_array_equal(sgd_linear_clf.predict(X), second_predictions[:, i])