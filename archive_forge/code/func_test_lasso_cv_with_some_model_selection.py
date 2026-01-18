import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_lasso_cv_with_some_model_selection():
    from sklearn import datasets
    from sklearn.model_selection import ShuffleSplit
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=ShuffleSplit(random_state=0)))
    pipe.fit(X, y)