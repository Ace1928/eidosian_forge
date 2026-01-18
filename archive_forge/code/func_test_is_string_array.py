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
def test_is_string_array(self):
    assert lib.is_string_array(np.array(['foo', 'bar']))
    assert not lib.is_string_array(np.array(['foo', 'bar', pd.NA], dtype=object), skipna=False)
    assert lib.is_string_array(np.array(['foo', 'bar', pd.NA], dtype=object), skipna=True)
    assert lib.is_string_array(np.array(['foo', 'bar', None], dtype=object), skipna=True)
    assert lib.is_string_array(np.array(['foo', 'bar', np.nan], dtype=object), skipna=True)
    assert not lib.is_string_array(np.array(['foo', 'bar', pd.NaT], dtype=object), skipna=True)
    assert not lib.is_string_array(np.array(['foo', 'bar', np.datetime64('NaT')], dtype=object), skipna=True)
    assert not lib.is_string_array(np.array(['foo', 'bar', Decimal('NaN')], dtype=object), skipna=True)
    assert not lib.is_string_array(np.array(['foo', 'bar', None], dtype=object), skipna=False)
    assert not lib.is_string_array(np.array(['foo', 'bar', np.nan], dtype=object), skipna=False)
    assert not lib.is_string_array(np.array([1, 2]))