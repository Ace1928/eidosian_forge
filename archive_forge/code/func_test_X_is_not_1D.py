import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('method', ['fit', 'fit_transform'])
@pytest.mark.parametrize('X', [[1, 2], np.array([3.0, 4.0])])
def test_X_is_not_1D(X, method):
    oh = OneHotEncoder()
    msg = 'Expected 2D array, got 1D array instead'
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)