import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype_class', [dict, Series])
def test_astype_dict_like(self, dtype_class):
    a = Series(date_range('2010-01-04', periods=5))
    b = Series(range(5))
    c = Series([0.0, 0.2, 0.4, 0.6, 0.8])
    d = Series(['1.0', '2', '3.14', '4', '5.4'])
    df = DataFrame({'a': a, 'b': b, 'c': c, 'd': d})
    original = df.copy(deep=True)
    dt1 = dtype_class({'b': 'str', 'd': 'float32'})
    result = df.astype(dt1)
    expected = DataFrame({'a': a, 'b': Series(['0', '1', '2', '3', '4'], dtype='object'), 'c': c, 'd': Series([1.0, 2.0, 3.14, 4.0, 5.4], dtype='float32')})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, original)
    dt2 = dtype_class({'b': np.float32, 'c': 'float32', 'd': np.float64})
    result = df.astype(dt2)
    expected = DataFrame({'a': a, 'b': Series([0.0, 1.0, 2.0, 3.0, 4.0], dtype='float32'), 'c': Series([0.0, 0.2, 0.4, 0.6, 0.8], dtype='float32'), 'd': Series([1.0, 2.0, 3.14, 4.0, 5.4], dtype='float64')})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, original)
    dt3 = dtype_class({'a': str, 'b': str, 'c': str, 'd': str})
    tm.assert_frame_equal(df.astype(dt3), df.astype(str))
    tm.assert_frame_equal(df, original)
    dt4 = dtype_class({'b': str, 2: str})
    dt5 = dtype_class({'e': str})
    msg_frame = "Only a column name can be used for the key in a dtype mappings argument. '{}' not found in columns."
    with pytest.raises(KeyError, match=msg_frame.format(2)):
        df.astype(dt4)
    with pytest.raises(KeyError, match=msg_frame.format('e')):
        df.astype(dt5)
    tm.assert_frame_equal(df, original)
    dt6 = dtype_class({col: df[col].dtype for col in df.columns})
    equiv = df.astype(dt6)
    tm.assert_frame_equal(df, equiv)
    tm.assert_frame_equal(df, original)
    dt7 = dtype_class({}) if dtype_class is dict else dtype_class({}, dtype=object)
    equiv = df.astype(dt7)
    tm.assert_frame_equal(df, equiv)
    tm.assert_frame_equal(df, original)