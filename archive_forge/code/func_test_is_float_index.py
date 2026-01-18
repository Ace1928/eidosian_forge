from statsmodels.compat.pandas import PD_LT_1_4, is_float_index, is_int_index
import numpy as np
import pandas as pd
import pytest
@pytest.mark.parametrize('float_size', [4, 8])
def test_is_float_index(float_size):
    index = pd.Index(np.arange(100.0), dtype=f'f{float_size}')
    assert is_float_index(index)
    assert not is_int_index(index)