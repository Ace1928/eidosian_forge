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
def test_multi_output_classification():
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    multi_target_forest = MultiOutputClassifier(forest)
    multi_target_forest.fit(X, y)
    predictions = multi_target_forest.predict(X)
    assert (n_samples, n_outputs) == predictions.shape
    predict_proba = multi_target_forest.predict_proba(X)
    assert len(predict_proba) == n_outputs
    for class_probabilities in predict_proba:
        assert (n_samples, n_classes) == class_probabilities.shape
    assert_array_equal(np.argmax(np.dstack(predict_proba), axis=1), predictions)
    for i in range(3):
        forest_ = clone(forest)
        forest_.fit(X, y[:, i])
        assert list(forest_.predict(X)) == list(predictions[:, i])
        assert_array_equal(list(forest_.predict_proba(X)), list(predict_proba[i]))