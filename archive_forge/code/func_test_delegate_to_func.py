import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_delegate_to_func():
    args_store = []
    kwargs_store = {}
    X = np.arange(10).reshape((5, 2))
    assert_array_equal(FunctionTransformer(_make_func(args_store, kwargs_store)).transform(X), X, 'transform should have returned X unchanged')
    assert args_store == [X], 'Incorrect positional arguments passed to func: {args}'.format(args=args_store)
    assert not kwargs_store, 'Unexpected keyword arguments passed to func: {args}'.format(args=kwargs_store)
    args_store[:] = []
    kwargs_store.clear()
    transformed = FunctionTransformer(_make_func(args_store, kwargs_store)).transform(X)
    assert_array_equal(transformed, X, err_msg='transform should have returned X unchanged')
    assert args_store == [X], 'Incorrect positional arguments passed to func: {args}'.format(args=args_store)
    assert not kwargs_store, 'Unexpected keyword arguments passed to func: {args}'.format(args=kwargs_store)