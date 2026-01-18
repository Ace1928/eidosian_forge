from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import (
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('stacker, feature_names, X, y, expected_names', [(StackingClassifier(estimators=[('lr', LogisticRegression(random_state=0)), ('svm', LinearSVC(dual='auto', random_state=0))]), iris.feature_names, X_iris, y_iris, ['stackingclassifier_lr0', 'stackingclassifier_lr1', 'stackingclassifier_lr2', 'stackingclassifier_svm0', 'stackingclassifier_svm1', 'stackingclassifier_svm2']), (StackingClassifier(estimators=[('lr', LogisticRegression(random_state=0)), ('other', 'drop'), ('svm', LinearSVC(dual='auto', random_state=0))]), iris.feature_names, X_iris[:100], y_iris[:100], ['stackingclassifier_lr', 'stackingclassifier_svm']), (StackingRegressor(estimators=[('lr', LinearRegression()), ('svm', LinearSVR(dual='auto', random_state=0))]), diabetes.feature_names, X_diabetes, y_diabetes, ['stackingregressor_lr', 'stackingregressor_svm'])], ids=['StackingClassifier_multiclass', 'StackingClassifier_binary', 'StackingRegressor'])
@pytest.mark.parametrize('passthrough', [True, False])
def test_get_feature_names_out(stacker, feature_names, X, y, expected_names, passthrough):
    """Check get_feature_names_out works for stacking."""
    stacker.set_params(passthrough=passthrough)
    stacker.fit(scale(X), y)
    if passthrough:
        expected_names = np.concatenate((expected_names, feature_names))
    names_out = stacker.get_feature_names_out(feature_names)
    assert_array_equal(names_out, expected_names)