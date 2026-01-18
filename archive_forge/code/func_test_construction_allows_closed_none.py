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
@pytest.mark.parametrize('subtype', ['interval[int64]', 'Interval[int64]', 'int64', np.dtype('int64')])
def test_construction_allows_closed_none(self, subtype):
    dtype = IntervalDtype(subtype)
    assert dtype.closed is None