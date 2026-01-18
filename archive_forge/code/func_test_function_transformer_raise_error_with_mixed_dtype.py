import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('X_type', ['array', 'series'])
def test_function_transformer_raise_error_with_mixed_dtype(X_type):
    """Check that `FunctionTransformer.check_inverse` raises error on mixed dtype."""
    mapping = {'one': 1, 'two': 2, 'three': 3, 5: 'five', 6: 'six'}
    inverse_mapping = {value: key for key, value in mapping.items()}
    dtype = 'object'
    data = ['one', 'two', 'three', 'one', 'one', 5, 6]
    data = _convert_container(data, X_type, columns_name=['value'], dtype=dtype)

    def func(X):
        return np.array([mapping[X[i]] for i in range(X.size)], dtype=object)

    def inverse_func(X):
        return _convert_container([inverse_mapping[x] for x in X], X_type, columns_name=['value'], dtype=dtype)
    transformer = FunctionTransformer(func=func, inverse_func=inverse_func, validate=False, check_inverse=True)
    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    with pytest.raises(ValueError, match=msg):
        transformer.fit(data)