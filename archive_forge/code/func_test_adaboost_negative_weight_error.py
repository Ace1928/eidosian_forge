import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('model, X, y', [(AdaBoostClassifier(), iris.data, iris.target), (AdaBoostRegressor(), diabetes.data, diabetes.target)])
def test_adaboost_negative_weight_error(model, X, y):
    sample_weight = np.ones_like(y)
    sample_weight[-1] = -10
    err_msg = 'Negative values in data passed to `sample_weight`'
    with pytest.raises(ValueError, match=err_msg):
        model.fit(X, y, sample_weight=sample_weight)