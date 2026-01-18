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
@pytest.mark.parametrize('val', [None, np.nan, float('nan')])
@pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
def test_maybe_convert_objects_nat_inference(self, val, dtype):
    dtype = np.dtype(dtype)
    vals = np.array([pd.NaT, val], dtype=object)
    result = lib.maybe_convert_objects(vals, convert_non_numeric=True, dtype_if_all_nat=dtype)
    assert result.dtype == dtype
    assert np.isnat(result).all()
    result = lib.maybe_convert_objects(vals[::-1], convert_non_numeric=True, dtype_if_all_nat=dtype)
    assert result.dtype == dtype
    assert np.isnat(result).all()