import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('in_dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('out_dtype', [None, np.float32, np.float64])
@pytest.mark.parametrize('encode', ['ordinal', 'onehot', 'onehot-dense'])
def test_consistent_dtype(in_dtype, out_dtype, encode):
    X_input = np.array(X, dtype=in_dtype)
    kbd = KBinsDiscretizer(n_bins=3, encode=encode, dtype=out_dtype)
    kbd.fit(X_input)
    if out_dtype is not None:
        expected_dtype = out_dtype
    elif out_dtype is None and X_input.dtype == np.float16:
        expected_dtype = np.float64
    else:
        expected_dtype = X_input.dtype
    Xt = kbd.transform(X_input)
    assert Xt.dtype == expected_dtype