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
@pytest.mark.parametrize(['est', 'pattern'], [(ColumnTransformer([('trans1', Trans(), [0]), ('trans2', Trans(), [1])], remainder=DoubleTrans()), '\\[ColumnTransformer\\].*\\(1 of 3\\) Processing trans1.* total=.*\\n\\[ColumnTransformer\\].*\\(2 of 3\\) Processing trans2.* total=.*\\n\\[ColumnTransformer\\].*\\(3 of 3\\) Processing remainder.* total=.*\\n$'), (ColumnTransformer([('trans1', Trans(), [0]), ('trans2', Trans(), [1])], remainder='passthrough'), '\\[ColumnTransformer\\].*\\(1 of 3\\) Processing trans1.* total=.*\\n\\[ColumnTransformer\\].*\\(2 of 3\\) Processing trans2.* total=.*\\n\\[ColumnTransformer\\].*\\(3 of 3\\) Processing remainder.* total=.*\\n$'), (ColumnTransformer([('trans1', Trans(), [0]), ('trans2', 'drop', [1])], remainder='passthrough'), '\\[ColumnTransformer\\].*\\(1 of 2\\) Processing trans1.* total=.*\\n\\[ColumnTransformer\\].*\\(2 of 2\\) Processing remainder.* total=.*\\n$'), (ColumnTransformer([('trans1', Trans(), [0]), ('trans2', 'passthrough', [1])], remainder='passthrough'), '\\[ColumnTransformer\\].*\\(1 of 3\\) Processing trans1.* total=.*\\n\\[ColumnTransformer\\].*\\(2 of 3\\) Processing trans2.* total=.*\\n\\[ColumnTransformer\\].*\\(3 of 3\\) Processing remainder.* total=.*\\n$'), (ColumnTransformer([('trans1', Trans(), [0])], remainder='passthrough'), '\\[ColumnTransformer\\].*\\(1 of 2\\) Processing trans1.* total=.*\\n\\[ColumnTransformer\\].*\\(2 of 2\\) Processing remainder.* total=.*\\n$'), (ColumnTransformer([('trans1', Trans(), [0]), ('trans2', Trans(), [1])], remainder='drop'), '\\[ColumnTransformer\\].*\\(1 of 2\\) Processing trans1.* total=.*\\n\\[ColumnTransformer\\].*\\(2 of 2\\) Processing trans2.* total=.*\\n$'), (ColumnTransformer([('trans1', Trans(), [0])], remainder='drop'), '\\[ColumnTransformer\\].*\\(1 of 1\\) Processing trans1.* total=.*\\n$')])
@pytest.mark.parametrize('method', ['fit', 'fit_transform'])
def test_column_transformer_verbose(est, pattern, method, capsys):
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    func = getattr(est, method)
    est.set_params(verbose=False)
    func(X_array)
    assert not capsys.readouterr().out, 'Got output for verbose=False'
    est.set_params(verbose=True)
    func(X_array)
    assert re.match(pattern, capsys.readouterr()[0])