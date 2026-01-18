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
def test_constructor_dtype_datetime64(self):
    s = Series(iNaT, dtype='M8[ns]', index=range(5))
    assert isna(s).all()
    s = Series(iNaT, index=range(5))
    assert not isna(s).all()
    s = Series(np.nan, dtype='M8[ns]', index=range(5))
    assert isna(s).all()
    s = Series([datetime(2001, 1, 2, 0, 0), iNaT], dtype='M8[ns]')
    assert isna(s[1])
    assert s.dtype == 'M8[ns]'
    s = Series([datetime(2001, 1, 2, 0, 0), np.nan], dtype='M8[ns]')
    assert isna(s[1])
    assert s.dtype == 'M8[ns]'