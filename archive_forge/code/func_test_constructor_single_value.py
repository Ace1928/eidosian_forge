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
def test_constructor_single_value(self):
    df = DataFrame(0.0, index=[1, 2, 3], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(df, DataFrame(np.zeros(df.shape).astype('float64'), df.index, df.columns))
    df = DataFrame(0, index=[1, 2, 3], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(df, DataFrame(np.zeros(df.shape).astype('int64'), df.index, df.columns))
    df = DataFrame('a', index=[1, 2], columns=['a', 'c'])
    tm.assert_frame_equal(df, DataFrame(np.array([['a', 'a'], ['a', 'a']], dtype=object), index=[1, 2], columns=['a', 'c']))
    msg = 'DataFrame constructor not properly called!'
    with pytest.raises(ValueError, match=msg):
        DataFrame('a', [1, 2])
    with pytest.raises(ValueError, match=msg):
        DataFrame('a', columns=['a', 'c'])
    msg = 'incompatible data and dtype'
    with pytest.raises(TypeError, match=msg):
        DataFrame('a', [1, 2], ['a', 'c'], float)