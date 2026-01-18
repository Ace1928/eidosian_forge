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
def test_bools(self):
    arr = np.array([True, False, True, True, True], dtype='O')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'boolean'
    arr = np.array([np.bool_(True), np.bool_(False)], dtype='O')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'boolean'
    arr = np.array([True, False, True, 'foo'], dtype='O')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'mixed'
    arr = np.array([True, False, True], dtype=bool)
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'boolean'
    arr = np.array([True, np.nan, False], dtype='O')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'boolean'
    result = lib.infer_dtype(arr, skipna=False)
    assert result == 'mixed'