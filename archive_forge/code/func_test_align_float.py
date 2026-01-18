from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_float(self, float_frame, using_copy_on_write):
    af, bf = float_frame.align(float_frame)
    assert af._mgr is not float_frame._mgr
    af, bf = float_frame.align(float_frame, copy=False)
    if not using_copy_on_write:
        assert af._mgr is float_frame._mgr
    else:
        assert af._mgr is not float_frame._mgr
    other = float_frame.iloc[:-5, :3]
    af, bf = float_frame.align(other, axis=0, fill_value=-1)
    tm.assert_index_equal(bf.columns, other.columns)
    join_idx = float_frame.index.join(other.index)
    diff_a = float_frame.index.difference(join_idx)
    diff_a_vals = af.reindex(diff_a).values
    assert (diff_a_vals == -1).all()
    af, bf = float_frame.align(other, join='right', axis=0)
    tm.assert_index_equal(bf.columns, other.columns)
    tm.assert_index_equal(bf.index, other.index)
    tm.assert_index_equal(af.index, other.index)
    other = float_frame.iloc[:-5, :3].copy()
    af, bf = float_frame.align(other, axis=1)
    tm.assert_index_equal(bf.columns, float_frame.columns)
    tm.assert_index_equal(bf.index, other.index)
    join_idx = float_frame.index.join(other.index)
    diff_a = float_frame.index.difference(join_idx)
    diff_a_vals = af.reindex(diff_a).values
    assert (diff_a_vals == -1).all()
    af, bf = float_frame.align(other, join='inner', axis=1)
    tm.assert_index_equal(bf.columns, other.columns)
    msg = "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        af, bf = float_frame.align(other, join='inner', axis=1, method='pad')
    tm.assert_index_equal(bf.columns, other.columns)
    msg = "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        af, bf = float_frame.align(other.iloc[:, 0], join='inner', axis=1, method=None, fill_value=None)
    tm.assert_index_equal(bf.index, Index([]).astype(bf.index.dtype))
    msg = "The 'method', 'limit', and 'fill_axis' keywords in DataFrame.align are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        af, bf = float_frame.align(other.iloc[:, 0], join='inner', axis=1, method=None, fill_value=0)
    tm.assert_index_equal(bf.index, Index([]).astype(bf.index.dtype))
    msg = 'No axis named 2 for object type DataFrame'
    with pytest.raises(ValueError, match=msg):
        float_frame.align(af.iloc[0, :3], join='inner', axis=2)