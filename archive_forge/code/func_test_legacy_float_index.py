from statsmodels.compat.pandas import PD_LT_1_4, is_float_index, is_int_index
import numpy as np
import pandas as pd
import pytest
@pytest.mark.skipif(not PD_LT_1_4, reason='Requires Float64Index')
def test_legacy_float_index():
    from pandas import Float64Index
    index = Float64Index(np.arange(100))
    assert not is_int_index(index)
    assert is_float_index(index)