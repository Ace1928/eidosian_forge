import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('selector', [[1], lambda x: [1], [False, True], lambda x: [False, True]])
def test_feature_names_out_non_pandas(selector):
    """Checks name when selecting the second column with numpy array"""
    X = [['a', 'z'], ['a', 'z'], ['b', 'z']]
    ct = ColumnTransformer([('ohe', OneHotEncoder(), selector)])
    ct.fit(X)
    assert_array_equal(ct.get_feature_names_out(), ['ohe__x1_z'])