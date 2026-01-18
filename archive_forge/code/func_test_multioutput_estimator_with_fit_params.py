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
@pytest.mark.filterwarnings('ignore:`n_features_in_` is deprecated')
@pytest.mark.parametrize('estimator, dataset', [(MultiOutputClassifier(DummyClassifierWithFitParams(strategy='prior')), datasets.make_multilabel_classification()), (MultiOutputRegressor(DummyRegressorWithFitParams()), datasets.make_regression(n_targets=3, random_state=0))])
def test_multioutput_estimator_with_fit_params(estimator, dataset):
    X, y = dataset
    some_param = np.zeros_like(X)
    estimator.fit(X, y, some_param=some_param)
    for dummy_estimator in estimator.estimators_:
        assert 'some_param' in dummy_estimator._fit_params