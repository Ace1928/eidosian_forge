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
@pytest.mark.parametrize('case', [np.array([2 ** 63, -1], dtype=object), np.array([str(2 ** 63), -1], dtype=object), np.array([str(2 ** 63), str(-1)], dtype=object), np.array([-1, 2 ** 63], dtype=object), np.array([-1, str(2 ** 63)], dtype=object), np.array([str(-1), str(2 ** 63)], dtype=object)])
@pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
def test_convert_numeric_int64_uint64(self, case, coerce, convert_to_masked_nullable):
    expected = case.astype(float) if coerce else case.copy()
    result, _ = lib.maybe_convert_numeric(case, set(), coerce_numeric=coerce, convert_to_masked_nullable=convert_to_masked_nullable)
    tm.assert_almost_equal(result, expected)