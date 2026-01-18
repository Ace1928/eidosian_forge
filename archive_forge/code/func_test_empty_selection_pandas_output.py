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
@pytest.mark.parametrize('empty_selection', [[], np.array([False, False]), [False, False]], ids=['list', 'bool', 'bool_int'])
def test_empty_selection_pandas_output(empty_selection):
    """Check that pandas output works when there is an empty selection.

    Non-regression test for gh-25487
    """
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame([[1.0, 2.2], [3.0, 1.0]], columns=['a', 'b'])
    ct = ColumnTransformer([('categorical', 'passthrough', empty_selection), ('numerical', StandardScaler(), ['a', 'b'])], verbose_feature_names_out=True)
    ct.set_output(transform='pandas')
    X_out = ct.fit_transform(X)
    assert_array_equal(X_out.columns, ['numerical__a', 'numerical__b'])
    ct.set_params(verbose_feature_names_out=False)
    X_out = ct.fit_transform(X)
    assert_array_equal(X_out.columns, ['a', 'b'])