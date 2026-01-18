import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['float64', 'int8', 'uint8', 'm8[ns]', 'M8[ns]'])
@pytest.mark.parametrize('dtype2', ['float64', 'int8', 'uint8', 'm8[ns]', 'M8[ns]'])
def test_concat_empty_series_dtypes_roundtrips(self, dtype, dtype2):
    if dtype == dtype2:
        pytest.skip('same dtype is not applicable for test')

    def int_result_type(dtype, dtype2):
        typs = {dtype.kind, dtype2.kind}
        if not len(typs - {'i', 'u', 'b'}) and (dtype.kind == 'i' or dtype2.kind == 'i'):
            return 'i'
        elif not len(typs - {'u', 'b'}) and (dtype.kind == 'u' or dtype2.kind == 'u'):
            return 'u'
        return None

    def float_result_type(dtype, dtype2):
        typs = {dtype.kind, dtype2.kind}
        if not len(typs - {'f', 'i', 'u'}) and (dtype.kind == 'f' or dtype2.kind == 'f'):
            return 'f'
        return None

    def get_result_type(dtype, dtype2):
        result = float_result_type(dtype, dtype2)
        if result is not None:
            return result
        result = int_result_type(dtype, dtype2)
        if result is not None:
            return result
        return 'O'
    dtype = np.dtype(dtype)
    dtype2 = np.dtype(dtype2)
    expected = get_result_type(dtype, dtype2)
    result = concat([Series(dtype=dtype), Series(dtype=dtype2)]).dtype
    assert result.kind == expected