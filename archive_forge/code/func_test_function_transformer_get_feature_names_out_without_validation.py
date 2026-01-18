import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_function_transformer_get_feature_names_out_without_validation():
    transformer = FunctionTransformer(feature_names_out='one-to-one', validate=False)
    X = np.random.rand(100, 2)
    transformer.fit_transform(X)
    names = transformer.get_feature_names_out(('a', 'b'))
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, ('a', 'b'))