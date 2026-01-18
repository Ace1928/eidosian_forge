import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('input_index', [None, ['a'], ['a', 'b']])
@pytest.mark.parametrize('keys', [['a'], ['a', 'b']])
@pytest.mark.parametrize('series', [True, False])
def test_groupby_dropna_with_multiindex_input(input_index, keys, series):
    obj = pd.DataFrame({'a': [1, np.nan], 'b': [1, 1], 'c': [2, 3]})
    expected = obj.set_index(keys)
    if series:
        expected = expected['c']
    elif input_index == ['a', 'b'] and keys == ['a']:
        expected = expected[['c']]
    if input_index is not None:
        obj = obj.set_index(input_index)
    gb = obj.groupby(keys, dropna=False)
    if series:
        gb = gb['c']
    result = gb.sum()
    tm.assert_equal(result, expected)