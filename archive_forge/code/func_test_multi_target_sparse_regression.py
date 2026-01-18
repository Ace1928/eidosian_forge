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
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + DOK_CONTAINERS + BSR_CONTAINERS)
def test_multi_target_sparse_regression(sparse_container):
    X, y = datasets.make_regression(n_targets=3, random_state=0)
    X_train, y_train = (X[:50], y[:50])
    X_test = X[50:]
    rgr = MultiOutputRegressor(Lasso(random_state=0))
    rgr_sparse = MultiOutputRegressor(Lasso(random_state=0))
    rgr.fit(X_train, y_train)
    rgr_sparse.fit(sparse_container(X_train), y_train)
    assert_almost_equal(rgr.predict(X_test), rgr_sparse.predict(sparse_container(X_test)))