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
@pytest.mark.parametrize('klass', [pd.array, Series, Index])
@pytest.mark.parametrize('skipna', [True, False])
def test_infer_dtype_period_array(self, klass, skipna):
    values = klass([Period('2011-01-01', freq='D'), Period('2011-01-02', freq='D'), pd.NaT])
    assert lib.infer_dtype(values, skipna=skipna) == 'period'
    values = klass([Period('2011-01-01', freq='D'), Period('2011-01-02', freq='M'), pd.NaT])
    exp = 'unknown-array' if klass is pd.array else 'mixed'
    assert lib.infer_dtype(values, skipna=skipna) == exp