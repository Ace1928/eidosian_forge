import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_kw_arg():
    X = np.linspace(0, 1, num=10).reshape((5, 2))
    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))
    assert_array_equal(F.transform(X), np.around(X, decimals=3))