from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('series, func, expected', chain(tm.get_cython_table_params(Series('a b c'.split()), [('mean', TypeError), ('prod', TypeError), ('std', TypeError), ('var', TypeError), ('median', TypeError), ('cumprod', TypeError)])))
def test_agg_cython_table_raises_series(series, func, expected, using_infer_string):
    msg = "[Cc]ould not convert|can't multiply sequence by non-int of type"
    if func == 'median' or func is np.nanmedian or func is np.median:
        msg = "Cannot convert \\['a' 'b' 'c'\\] to numeric"
    if using_infer_string:
        import pyarrow as pa
        expected = (expected, pa.lib.ArrowNotImplementedError)
    msg = msg + '|does not support|has no kernel'
    warn = None if isinstance(func, str) else FutureWarning
    with pytest.raises(expected, match=msg):
        with tm.assert_produces_warning(warn, match='is currently using Series.*'):
            series.agg(func)