from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'US/Central'])
def test_astype_array_fallback(self, tz):
    obj = date_range('2000', periods=2, tz=tz, name='idx')
    result = obj.astype(bool)
    expected = Index(np.array([True, True]), name='idx')
    tm.assert_index_equal(result, expected)
    result = obj._data.astype(bool)
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)