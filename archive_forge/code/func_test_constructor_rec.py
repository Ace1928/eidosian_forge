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
def test_constructor_rec(self, float_frame):
    rec = float_frame.to_records(index=False)
    rec.dtype.names = list(rec.dtype.names)[::-1]
    index = float_frame.index
    df = DataFrame(rec)
    tm.assert_index_equal(df.columns, Index(rec.dtype.names))
    df2 = DataFrame(rec, index=index)
    tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
    tm.assert_index_equal(df2.index, index)
    rng = np.arange(len(rec))[::-1]
    df3 = DataFrame(rec, index=rng, columns=['C', 'B'])
    expected = DataFrame(rec, index=rng).reindex(columns=['C', 'B'])
    tm.assert_frame_equal(df3, expected)