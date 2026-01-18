from re import escape
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn import datasets, svm
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import check_classification_targets, type_of_target
def test_ecoc_gridsearch():
    ecoc = OutputCodeClassifier(LinearSVC(dual='auto', random_state=0), random_state=0)
    Cs = [0.1, 0.5, 0.8]
    cv = GridSearchCV(ecoc, {'estimator__C': Cs})
    cv.fit(iris.data, iris.target)
    best_C = cv.best_estimator_.estimators_[0].C
    assert best_C in Cs