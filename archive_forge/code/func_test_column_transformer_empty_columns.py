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
@pytest.mark.parametrize('pandas', [True, False], ids=['pandas', 'numpy'])
@pytest.mark.parametrize('column_selection', [[], np.array([False, False]), [False, False]], ids=['list', 'bool', 'bool_int'])
@pytest.mark.parametrize('callable_column', [False, True])
def test_column_transformer_empty_columns(pandas, column_selection, callable_column):
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_both = X_array
    if pandas:
        pd = pytest.importorskip('pandas')
        X = pd.DataFrame(X_array, columns=['first', 'second'])
    else:
        X = X_array
    if callable_column:
        column = lambda X: column_selection
    else:
        column = column_selection
    ct = ColumnTransformer([('trans1', Trans(), [0, 1]), ('trans2', TransRaise(), column)])
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[1][1], TransRaise)
    ct = ColumnTransformer([('trans1', TransRaise(), column), ('trans2', Trans(), [0, 1])])
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[0][1], TransRaise)
    ct = ColumnTransformer([('trans', TransRaise(), column)], remainder='passthrough')
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[0][1], TransRaise)
    fixture = np.array([[], [], []])
    ct = ColumnTransformer([('trans', TransRaise(), column)], remainder='drop')
    assert_array_equal(ct.fit_transform(X), fixture)
    assert_array_equal(ct.fit(X).transform(X), fixture)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[0][1], TransRaise)