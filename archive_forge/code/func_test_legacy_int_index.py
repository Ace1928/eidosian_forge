from statsmodels.compat.pandas import PD_LT_1_4, is_float_index, is_int_index
import numpy as np
import pandas as pd
import pytest
@pytest.mark.skipif(not PD_LT_1_4, reason='Requires U/Int64Index')
def test_legacy_int_index():
    from pandas import Int64Index, UInt64Index
    index = Int64Index(np.arange(100))
    assert is_int_index(index)
    assert not is_float_index(index)
    index = UInt64Index(np.arange(100))
    assert is_int_index(index)
    assert not is_float_index(index)