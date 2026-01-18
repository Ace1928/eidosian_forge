from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_no_unnecessary_upcast(float_numpy_dtype, frame_or_series):
    dims = frame_or_series(dtype=object).ndim
    dt = float_numpy_dtype
    dfs = [frame_or_series(np.array([1], dtype=dt, ndmin=dims)), frame_or_series(np.array([np.nan], dtype=dt, ndmin=dims)), frame_or_series(np.array([5], dtype=dt, ndmin=dims))]
    x = concat(dfs)
    assert x.values.dtype == dt