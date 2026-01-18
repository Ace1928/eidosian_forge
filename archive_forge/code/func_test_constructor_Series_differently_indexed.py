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
def test_constructor_Series_differently_indexed(self):
    s1 = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
    s2 = Series([1, 2, 3], index=['a', 'b', 'c'])
    other_index = Index(['a', 'b'])
    df1 = DataFrame(s1, index=other_index)
    exp1 = DataFrame(s1.reindex(other_index))
    assert df1.columns[0] == 'x'
    tm.assert_frame_equal(df1, exp1)
    df2 = DataFrame(s2, index=other_index)
    exp2 = DataFrame(s2.reindex(other_index))
    assert df2.columns[0] == 0
    tm.assert_index_equal(df2.index, other_index)
    tm.assert_frame_equal(df2, exp2)