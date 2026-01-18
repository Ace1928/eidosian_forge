from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_convert_objects_leave_decimal_alone():
    s = Series(range(5))
    labels = np.array(['a', 'b', 'c', 'd', 'e'], dtype='O')

    def convert_fast(x):
        return Decimal(str(x.mean()))

    def convert_force_pure(x):
        assert len(x.values.base) > 0
        return Decimal(str(x.mean()))
    grouped = s.groupby(labels)
    result = grouped.agg(convert_fast)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)
    result = grouped.agg(convert_force_pure)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)