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
def test_constructor_dtype_datetime64_11(self):
    pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
    dates = [np.datetime64(x) for x in pydates]
    dts = Series(dates, dtype='datetime64[ns]')
    dts.astype('int64')
    msg = 'Converting from datetime64\\[ns\\] to int32 is not supported'
    with pytest.raises(TypeError, match=msg):
        dts.astype('int32')
    result = Series(dts, dtype=np.int64)
    expected = Series(dts.astype(np.int64))
    tm.assert_series_equal(result, expected)