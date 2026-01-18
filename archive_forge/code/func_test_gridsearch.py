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
def test_gridsearch():
    """Check GridSearch support."""
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=3)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    params = {'lr__C': [1.0, 100.0], 'voting': ['soft', 'hard'], 'weights': [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]]}
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=2)
    grid.fit(X_scaled, y)