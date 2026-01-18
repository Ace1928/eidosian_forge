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
@pytest.mark.parametrize('subtype', [None, 'interval', 'Interval'])
def test_construction_generic(self, subtype):
    i = IntervalDtype(subtype)
    assert i.subtype is None
    msg = 'is_interval_dtype is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert is_interval_dtype(i)