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
def test_constructor_2d_index(self):
    df = DataFrame([[1]], columns=[[1]], index=[1, 2])
    expected = DataFrame([1, 1], index=Index([1, 2], dtype='int64'), columns=MultiIndex(levels=[[1]], codes=[[0]]))
    tm.assert_frame_equal(df, expected)
    df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
    expected = DataFrame([1, 1], index=MultiIndex(levels=[[1, 2]], codes=[[0, 1]]), columns=MultiIndex(levels=[[1]], codes=[[0]]))
    tm.assert_frame_equal(df, expected)