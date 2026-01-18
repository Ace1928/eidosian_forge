from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_with_datetime_tz2(self):
    ser = Series(NaT, index=[0, 1], dtype='datetime64[ns, US/Eastern]')
    dti = DatetimeIndex(['NaT', 'NaT'], tz='US/Eastern').as_unit('ns')
    expected = Series(dti)
    tm.assert_series_equal(ser, expected)