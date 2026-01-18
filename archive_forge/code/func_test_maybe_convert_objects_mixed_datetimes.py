import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_maybe_convert_objects_mixed_datetimes(self):
    ts = Timestamp('now')
    vals = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]
    for data in itertools.permutations(vals):
        data = np.array(list(data), dtype=object)
        expected = DatetimeIndex(data)._data._ndarray
        result = lib.maybe_convert_objects(data, convert_non_numeric=True)
        tm.assert_numpy_array_equal(result, expected)