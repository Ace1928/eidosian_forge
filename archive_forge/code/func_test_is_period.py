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
def test_is_period(self):
    msg = 'is_period is deprecated and will be removed in a future version'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert lib.is_period(Period('2011-01', freq='M'))
        assert not lib.is_period(PeriodIndex(['2011-01'], freq='M'))
        assert not lib.is_period(Timestamp('2011-01'))
        assert not lib.is_period(1)
        assert not lib.is_period(np.nan)