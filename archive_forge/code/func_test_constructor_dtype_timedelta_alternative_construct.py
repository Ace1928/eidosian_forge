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
def test_constructor_dtype_timedelta_alternative_construct(self):
    result = Series([1000000, 200000, 3000000], dtype='timedelta64[ns]')
    expected = Series(pd.to_timedelta([1000000, 200000, 3000000], unit='ns'))
    tm.assert_series_equal(result, expected)