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
@pytest.mark.parametrize('arr', [np.array([Timestamp('2011-01-01'), Timestamp('2011-01-02')]), np.array([datetime(2011, 1, 1), datetime(2012, 2, 1)]), np.array([datetime(2011, 1, 1), Timestamp('2011-01-02')])])
def test_infer_dtype_datetime(self, arr):
    assert lib.infer_dtype(arr, skipna=True) == 'datetime'