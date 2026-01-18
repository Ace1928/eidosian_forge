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
@pytest.mark.parametrize('exp', [IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True])), IntegerArray(np.array([2, 0], dtype='int64'), np.array([False, True]))])
def test_maybe_convert_objects_nullable_integer(self, exp):
    arr = np.array([2, np.nan], dtype=object)
    result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
    tm.assert_extension_array_equal(result, exp)