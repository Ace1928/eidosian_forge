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
@pytest.mark.parametrize('container_type', ['list', 'array', 'dataframe'])
def test_multilabel(container_type):
    """Check if error is raised for multilabel classification."""
    X, y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=False, random_state=123)
    y = _convert_container(y, container_type)
    clf = OneVsRestClassifier(SVC(kernel='linear'))
    eclf = VotingClassifier(estimators=[('ovr', clf)], voting='hard')
    err_msg = 'only supports binary or multiclass classification'
    with pytest.raises(NotImplementedError, match=err_msg):
        eclf.fit(X, y)