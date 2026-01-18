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
def test_is_scalar_builtin_scalars(self):
    assert is_scalar(None)
    assert is_scalar(True)
    assert is_scalar(False)
    assert is_scalar(Fraction())
    assert is_scalar(0.0)
    assert is_scalar(1)
    assert is_scalar(complex(2))
    assert is_scalar(float('NaN'))
    assert is_scalar(np.nan)
    assert is_scalar('foobar')
    assert is_scalar(b'foobar')
    assert is_scalar(datetime(2014, 1, 1))
    assert is_scalar(date(2014, 1, 1))
    assert is_scalar(time(12, 0))
    assert is_scalar(timedelta(hours=1))
    assert is_scalar(pd.NaT)
    assert is_scalar(pd.NA)