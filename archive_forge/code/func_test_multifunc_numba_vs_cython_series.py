import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('agg_kwargs', [{'func': ['min', 'max']}, {'func': 'min'}, {'min_val': 'min', 'max_val': 'max'}])
def test_multifunc_numba_vs_cython_series(agg_kwargs):
    pytest.importorskip('numba')
    labels = ['a', 'a', 'b', 'b', 'a']
    data = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    grouped = data.groupby(labels)
    agg_kwargs['engine'] = 'numba'
    result = grouped.agg(**agg_kwargs)
    agg_kwargs['engine'] = 'cython'
    expected = grouped.agg(**agg_kwargs)
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)