from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('obj', [tm.SubclassedDataFrame({'A': np.arange(0, 10)}), tm.SubclassedSeries(np.arange(0, 10), name='A')])
def test_groupby_preserves_subclass(obj, groupby_func):
    if isinstance(obj, Series) and groupby_func in {'corrwith'}:
        pytest.skip(f'Not applicable for Series and {groupby_func}')
    grouped = obj.groupby(np.arange(0, 10))
    assert isinstance(grouped.get_group(0), type(obj))
    args = get_groupby_method_args(groupby_func, obj)
    warn = FutureWarning if groupby_func == 'fillna' else None
    msg = f'{type(grouped).__name__}.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=msg, raise_on_extra_warnings=False):
        result1 = getattr(grouped, groupby_func)(*args)
    with tm.assert_produces_warning(warn, match=msg, raise_on_extra_warnings=False):
        result2 = grouped.agg(groupby_func, *args)
    slices = {'ngroup', 'cumcount', 'size'}
    if isinstance(obj, DataFrame) and groupby_func in slices:
        assert isinstance(result1, tm.SubclassedSeries)
    else:
        assert isinstance(result1, type(obj))
    if isinstance(result1, DataFrame):
        tm.assert_frame_equal(result1, result2)
    else:
        tm.assert_series_equal(result1, result2)