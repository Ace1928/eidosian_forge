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
def test_decimals(self):
    arr = np.array([Decimal(1), Decimal(2), Decimal(3)])
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'decimal'
    arr = np.array([1.0, 2.0, Decimal(3)])
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'mixed'
    result = lib.infer_dtype(arr[::-1], skipna=True)
    assert result == 'mixed'
    arr = np.array([Decimal(1), Decimal('NaN'), Decimal(3)])
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'decimal'
    arr = np.array([Decimal(1), np.nan, Decimal(3)], dtype='O')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'decimal'