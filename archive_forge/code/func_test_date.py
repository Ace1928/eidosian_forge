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
def test_date(self):
    dates = [date(2012, 1, day) for day in range(1, 20)]
    index = Index(dates)
    assert index.inferred_type == 'date'
    dates = [date(2012, 1, day) for day in range(1, 20)] + [np.nan]
    result = lib.infer_dtype(dates, skipna=False)
    assert result == 'mixed'
    result = lib.infer_dtype(dates, skipna=True)
    assert result == 'date'