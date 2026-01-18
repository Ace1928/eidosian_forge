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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_ecoc_delegate_sparse_base_estimator(csc_container):
    X, y = (iris.data, iris.target)
    X_sp = csc_container(X)
    base_estimator = CheckingClassifier(check_X=check_array, check_X_params={'ensure_2d': True, 'accept_sparse': False})
    ecoc = OutputCodeClassifier(base_estimator, random_state=0)
    with pytest.raises(TypeError, match='Sparse data was passed'):
        ecoc.fit(X_sp, y)
    ecoc.fit(X, y)
    with pytest.raises(TypeError, match='Sparse data was passed'):
        ecoc.predict(X_sp)
    ecoc = OutputCodeClassifier(LinearSVC(dual='auto', random_state=0))
    ecoc.fit(X_sp, y).predict(X_sp)
    assert len(ecoc.estimators_) == 4