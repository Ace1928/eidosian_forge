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
def test_multiclass_multioutput_estimator():
    svc = LinearSVC(dual='auto', random_state=0)
    multi_class_svc = OneVsRestClassifier(svc)
    multi_target_svc = MultiOutputClassifier(multi_class_svc)
    multi_target_svc.fit(X, y)
    predictions = multi_target_svc.predict(X)
    assert (n_samples, n_outputs) == predictions.shape
    for i in range(3):
        multi_class_svc_ = clone(multi_class_svc)
        multi_class_svc_.fit(X, y[:, i])
        assert list(multi_class_svc_.predict(X)) == list(predictions[:, i])