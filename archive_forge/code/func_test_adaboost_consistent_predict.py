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
@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_adaboost_consistent_predict(algorithm):
    X_train, X_test, y_train, y_test = train_test_split(*datasets.load_digits(return_X_y=True), random_state=42)
    model = AdaBoostClassifier(algorithm=algorithm, random_state=42)
    model.fit(X_train, y_train)
    assert_array_equal(np.argmax(model.predict_proba(X_test), axis=1), model.predict(X_test))