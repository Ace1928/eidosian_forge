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
def test_is_scalar_numpy_array_scalars(self):
    assert is_scalar(np.int64(1))
    assert is_scalar(np.float64(1.0))
    assert is_scalar(np.int32(1))
    assert is_scalar(np.complex64(2))
    assert is_scalar(np.object_('foobar'))
    assert is_scalar(np.str_('foobar'))
    assert is_scalar(np.bytes_(b'foobar'))
    assert is_scalar(np.datetime64('2014-01-01'))
    assert is_scalar(np.timedelta64(1, 'h'))