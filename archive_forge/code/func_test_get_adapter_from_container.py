import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_get_adapter_from_container():
    """Check the behavior fo `_get_adapter_from_container`."""
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 100]})
    adapter = _get_adapter_from_container(X)
    assert adapter.container_lib == 'pandas'
    err_msg = 'The container does not have a registered adapter in scikit-learn.'
    with pytest.raises(ValueError, match=err_msg):
        _get_adapter_from_container(X.to_numpy())