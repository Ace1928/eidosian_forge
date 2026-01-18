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
@pytest.mark.parametrize('values', [[date(2020, 1, 1), Timestamp('2020-01-01')], [Timestamp('2020-01-01'), date(2020, 1, 1)], [date(2020, 1, 1), pd.NaT], [pd.NaT, date(2020, 1, 1)]])
@pytest.mark.parametrize('skipna', [True, False])
def test_infer_dtype_date_order_invariant(self, values, skipna):
    result = lib.infer_dtype(values, skipna=skipna)
    assert result == 'date'