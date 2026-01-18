import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('other', [[datetime.timedelta(1), datetime.timedelta(2)], [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)], [pd.Period('2000'), pd.Period('2001')], ['a', 'b']], ids=['timedelta', 'datetime', 'period', 'object'])
def test_index_ops_defer_to_unknown_subclasses(other):
    values = np.array([datetime.date(2000, 1, 1), datetime.date(2000, 1, 2)], dtype=object)
    a = MyIndex._simple_new(values)
    other = pd.Index(other)
    result = other + a
    assert isinstance(result, MyIndex)
    assert a._calls == 1