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
@pytest.mark.parametrize('subtype', ['int64', 'uint64', 'float64', 'complex128', 'datetime64', 'timedelta64', PeriodDtype('Q')])
def test_name_repr(self, subtype):
    closed = 'right' if subtype is not None else None
    dtype = IntervalDtype(subtype, closed=closed)
    expected = f'interval[{subtype}, {closed}]'
    assert str(dtype) == expected
    assert dtype.name == 'interval'