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
def test_infer_dtype_period_mixed(self):
    arr = np.array([Period('2011-01', freq='M'), np.datetime64('nat')], dtype=object)
    assert lib.infer_dtype(arr, skipna=False) == 'mixed'
    arr = np.array([np.datetime64('nat'), Period('2011-01', freq='M')], dtype=object)
    assert lib.infer_dtype(arr, skipna=False) == 'mixed'