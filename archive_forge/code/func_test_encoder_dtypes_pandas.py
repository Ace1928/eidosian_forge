import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_encoder_dtypes_pandas():
    pd = pytest.importorskip('pandas')
    enc = OneHotEncoder(categories='auto')
    exp = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]], dtype='float64')
    X = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}, dtype='int64')
    enc.fit(X)
    assert all([enc.categories_[i].dtype == 'int64' for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)
    X = pd.DataFrame({'A': [1, 2], 'B': ['a', 'b'], 'C': [3.0, 4.0]})
    X_type = [X['A'].dtype, X['B'].dtype, X['C'].dtype]
    enc.fit(X)
    assert all([enc.categories_[i].dtype == X_type[i] for i in range(3)])
    assert_array_equal(enc.transform(X).toarray(), exp)