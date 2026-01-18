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
def test_constructor(self, datetime_series, using_infer_string):
    empty_series = Series()
    assert datetime_series.index._is_all_dates
    derived = Series(datetime_series)
    assert derived.index._is_all_dates
    tm.assert_index_equal(derived.index, datetime_series.index)
    assert id(datetime_series.index) == id(derived.index)
    mixed = Series(['hello', np.nan], index=[0, 1])
    assert mixed.dtype == np.object_ if not using_infer_string else 'string'
    assert np.isnan(mixed[1])
    assert not empty_series.index._is_all_dates
    assert not Series().index._is_all_dates
    with pytest.raises(ValueError, match='Data must be 1-dimensional, got ndarray of shape \\(3, 3\\) instead'):
        Series(np.random.default_rng(2).standard_normal((3, 3)), index=np.arange(3))
    mixed.name = 'Series'
    rs = Series(mixed).name
    xp = 'Series'
    assert rs == xp
    m = MultiIndex.from_arrays([[1, 2], [3, 4]])
    msg = 'initializing a Series from a MultiIndex is not supported'
    with pytest.raises(NotImplementedError, match=msg):
        Series(m)