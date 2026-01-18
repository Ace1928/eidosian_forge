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
@pytest.mark.parametrize('string', ['foo', 'foo[int64]', 'IntervalA'])
def test_construction_from_string_error_subtype(self, string):
    msg = 'Incorrectly formatted string passed to constructor. Valid formats include Interval or Interval\\[dtype\\] where dtype is numeric, datetime, or timedelta'
    with pytest.raises(TypeError, match=msg):
        IntervalDtype.construct_from_string(string)