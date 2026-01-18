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
@pytest.mark.parametrize('estimator', [RandomForestClassifier(n_estimators=2), MultiOutputClassifier(RandomForestClassifier(n_estimators=2)), ClassifierChain(RandomForestClassifier(n_estimators=2))])
def test_multi_output_classes_(estimator):
    estimator.fit(X, y)
    assert isinstance(estimator.classes_, list)
    assert len(estimator.classes_) == n_outputs
    for estimator_classes, expected_classes in zip(classes, estimator.classes_):
        assert_array_equal(estimator_classes, expected_classes)