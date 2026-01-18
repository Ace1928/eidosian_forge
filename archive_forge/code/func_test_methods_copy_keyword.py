import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('copy', [True, None, False])
@pytest.mark.parametrize('method', [lambda df, copy: df.rename(columns=str.lower, copy=copy), lambda df, copy: df.reindex(columns=['a', 'c'], copy=copy), lambda df, copy: df.reindex_like(df, copy=copy), lambda df, copy: df.align(df, copy=copy)[0], lambda df, copy: df.set_axis(['a', 'b', 'c'], axis='index', copy=copy), lambda df, copy: df.rename_axis(index='test', copy=copy), lambda df, copy: df.rename_axis(columns='test', copy=copy), lambda df, copy: df.astype({'b': 'int64'}, copy=copy), lambda df, copy: df.swapaxes(0, 0, copy=copy), lambda df, copy: df.truncate(0, 5, copy=copy), lambda df, copy: df.infer_objects(copy=copy), lambda df, copy: df.to_timestamp(copy=copy), lambda df, copy: df.to_period(freq='D', copy=copy), lambda df, copy: df.tz_localize('US/Central', copy=copy), lambda df, copy: df.tz_convert('US/Central', copy=copy), lambda df, copy: df.set_flags(allows_duplicate_labels=False, copy=copy)], ids=['rename', 'reindex', 'reindex_like', 'align', 'set_axis', 'rename_axis0', 'rename_axis1', 'astype', 'swapaxes', 'truncate', 'infer_objects', 'to_timestamp', 'to_period', 'tz_localize', 'tz_convert', 'set_flags'])
def test_methods_copy_keyword(request, method, copy, using_copy_on_write, using_array_manager):
    index = None
    if 'to_timestamp' in request.node.callspec.id:
        index = period_range('2012-01-01', freq='D', periods=3)
    elif 'to_period' in request.node.callspec.id:
        index = date_range('2012-01-01', freq='D', periods=3)
    elif 'tz_localize' in request.node.callspec.id:
        index = date_range('2012-01-01', freq='D', periods=3)
    elif 'tz_convert' in request.node.callspec.id:
        index = date_range('2012-01-01', freq='D', periods=3, tz='Europe/Brussels')
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]}, index=index)
    if 'swapaxes' in request.node.callspec.id:
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df2 = method(df, copy=copy)
    else:
        df2 = method(df, copy=copy)
    share_memory = using_copy_on_write or copy is False
    if request.node.callspec.id.startswith('reindex-'):
        if not using_copy_on_write and (not using_array_manager) and (copy is False):
            share_memory = False
    if share_memory:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))