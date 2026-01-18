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
def test_X_is_not_1D_pandas(method):
    pd = pytest.importorskip('pandas')
    X = pd.Series([6, 3, 4, 6])
    oh = OneHotEncoder()
    msg = f'Expected a 2-dimensional container but got {type(X)} instead.'
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)