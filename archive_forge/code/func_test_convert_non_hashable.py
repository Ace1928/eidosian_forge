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
def test_convert_non_hashable(self):
    arr = np.array([[10.0, 2], 1.0, 'apple'], dtype=object)
    result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
    tm.assert_numpy_array_equal(result, np.array([np.nan, 1.0, np.nan]))