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
def test_fromValue(self, datetime_series, using_infer_string):
    nans = Series(np.nan, index=datetime_series.index, dtype=np.float64)
    assert nans.dtype == np.float64
    assert len(nans) == len(datetime_series)
    strings = Series('foo', index=datetime_series.index)
    assert strings.dtype == np.object_ if not using_infer_string else 'string'
    assert len(strings) == len(datetime_series)
    d = datetime.now()
    dates = Series(d, index=datetime_series.index)
    assert dates.dtype == 'M8[us]'
    assert len(dates) == len(datetime_series)
    categorical = Series(0, index=datetime_series.index, dtype='category')
    expected = Series(0, index=datetime_series.index).astype('category')
    assert categorical.dtype == 'category'
    assert len(categorical) == len(datetime_series)
    tm.assert_series_equal(categorical, expected)