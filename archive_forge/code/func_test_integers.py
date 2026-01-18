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
def test_integers(self):
    arr = np.array([1, 2, 3, np.int64(4), np.int32(5)], dtype='O')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'integer'
    arr = np.array([1, 2, 3, np.int64(4), np.int32(5), 'foo'], dtype='O')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'mixed-integer'
    arr = np.array([1, 2, 3, 4, 5], dtype='i4')
    result = lib.infer_dtype(arr, skipna=True)
    assert result == 'integer'