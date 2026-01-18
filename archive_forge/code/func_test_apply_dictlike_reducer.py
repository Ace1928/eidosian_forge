import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('ops', [{'A': np.sum}, {'A': np.sum, 'B': np.mean}, Series({'A': np.sum}), Series({'A': np.sum, 'B': np.mean})])
@pytest.mark.parametrize('how, kwargs', [['agg', {}], ['apply', {'by_row': 'compat'}], ['apply', {'by_row': False}]])
def test_apply_dictlike_reducer(string_series, ops, how, kwargs, by_row):
    expected = Series({name: op(string_series) for name, op in ops.items()})
    expected.name = string_series.name
    warn = FutureWarning if how == 'agg' else None
    msg = 'using Series.[sum|mean]'
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)