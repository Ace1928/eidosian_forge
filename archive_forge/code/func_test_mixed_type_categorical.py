import warnings
import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import (
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tests.test_tree import assert_is_subtree
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state
def test_mixed_type_categorical():
    """Check that we raise a proper error when a column has mixed types and
    the sorting of `np.unique` will fail."""
    X = np.array(['A', 'B', 'C', np.nan], dtype=object).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])
    from sklearn.preprocessing import OrdinalEncoder
    clf = make_pipeline(OrdinalEncoder(encoded_missing_value=-1), LogisticRegression()).fit(X, y)
    with pytest.raises(ValueError, match='The column #0 contains mixed data types'):
        partial_dependence(clf, X, features=[0])