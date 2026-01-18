import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_construction_non_nanosecond(self):
    res = DatetimeTZDtype('ms', 'US/Eastern')
    assert res.unit == 'ms'
    assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
    assert res.str == '|M8[ms]'
    assert str(res) == 'datetime64[ms, US/Eastern]'
    assert res.base == np.dtype('M8[ms]')