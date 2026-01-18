import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test__row_tensor_product_errors():
    import pytest
    pytest.raises(ValueError, _row_tensor_product, [])
    pytest.raises(ValueError, _row_tensor_product, [np.arange(1, 5)])
    pytest.raises(ValueError, _row_tensor_product, [np.arange(1, 5), np.arange(1, 5)])
    pytest.raises(ValueError, _row_tensor_product, [np.arange(1, 13).reshape((3, 4)), np.arange(1, 13).reshape((4, 3))])