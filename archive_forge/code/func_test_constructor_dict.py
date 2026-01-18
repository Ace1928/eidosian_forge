import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_dict(self):
    datetime_series = Series(np.arange(30, dtype=np.float64), index=date_range('2020-01-01', periods=30))
    datetime_series_short = datetime_series[5:]
    frame = DataFrame({'col1': datetime_series, 'col2': datetime_series_short})
    assert len(datetime_series) == 30
    assert len(datetime_series_short) == 25
    tm.assert_series_equal(frame['col1'], datetime_series.rename('col1'))
    exp = Series(np.concatenate([[np.nan] * 5, datetime_series_short.values]), index=datetime_series.index, name='col2')
    tm.assert_series_equal(exp, frame['col2'])
    frame = DataFrame({'col1': datetime_series, 'col2': datetime_series_short}, columns=['col2', 'col3', 'col4'])
    assert len(frame) == len(datetime_series_short)
    assert 'col1' not in frame
    assert isna(frame['col3']).all()
    assert len(DataFrame()) == 0
    msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
    with pytest.raises(ValueError, match=msg):
        DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})