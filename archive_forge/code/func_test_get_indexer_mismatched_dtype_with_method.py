from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['pad', 'backfill', 'nearest'])
def test_get_indexer_mismatched_dtype_with_method(self, non_comparable_idx, method):
    dti = date_range('2016-01-01', periods=3)
    pi = dti.to_period('D')
    other = non_comparable_idx
    msg = re.escape(f'Cannot compare dtypes {pi.dtype} and {other.dtype}')
    with pytest.raises(TypeError, match=msg):
        pi.get_indexer(other, method=method)
    for dtype in ['object', 'category']:
        other2 = other.astype(dtype)
        if dtype == 'object' and isinstance(other, PeriodIndex):
            continue
        msg = '|'.join([re.escape(msg) for msg in (f'Cannot compare dtypes {pi.dtype} and {other.dtype}', ' not supported between instances of ')])
        with pytest.raises(TypeError, match=msg):
            pi.get_indexer(other2, method=method)