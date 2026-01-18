import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_multilabel_classification
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
def test_get_features_names_out_regressor():
    """Check get_feature_names_out output for regressor."""
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]
    voting = VotingRegressor(estimators=[('lr', LinearRegression()), ('tree', DecisionTreeRegressor(random_state=0)), ('ignore', 'drop')])
    voting.fit(X, y)
    names_out = voting.get_feature_names_out()
    expected_names = ['votingregressor_lr', 'votingregressor_tree']
    assert_array_equal(names_out, expected_names)