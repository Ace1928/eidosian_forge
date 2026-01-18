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
def test_adaboostclassifier_without_sample_weight(algorithm):
    X, y = (iris.data, iris.target)
    estimator = NoSampleWeightWrapper(DummyClassifier())
    clf = AdaBoostClassifier(estimator=estimator, algorithm=algorithm)
    err_msg = "{} doesn't support sample_weight".format(estimator.__class__.__name__)
    with pytest.raises(ValueError, match=err_msg):
        clf.fit(X, y)