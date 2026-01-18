from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_coerce_float_valid(self, float_numpy_dtype):
    s = Series([1, 2, 3.5], dtype=float_numpy_dtype)
    expected = Series([1, 2, 3.5]).astype(float_numpy_dtype)
    tm.assert_series_equal(s, expected)