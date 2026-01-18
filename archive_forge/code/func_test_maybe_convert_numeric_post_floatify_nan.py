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
@pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
def test_maybe_convert_numeric_post_floatify_nan(self, coerce, convert_to_masked_nullable):
    data = np.array(['1.200', '-999.000', '4.500'], dtype=object)
    expected = np.array([1.2, np.nan, 4.5], dtype=np.float64)
    nan_values = {-999, -999.0}
    out = lib.maybe_convert_numeric(data, nan_values, coerce, convert_to_masked_nullable=convert_to_masked_nullable)
    if convert_to_masked_nullable:
        expected = FloatingArray(expected, np.isnan(expected))
        tm.assert_extension_array_equal(expected, FloatingArray(*out))
    else:
        out = out[0]
        tm.assert_numpy_array_equal(out, expected)