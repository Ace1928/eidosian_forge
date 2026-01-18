from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'Asia/Tokyo'])
def test_astype_object_tz(self, tz):
    idx = date_range(start='2013-01-01', periods=4, freq='ME', name='idx', tz=tz)
    expected_list = [Timestamp('2013-01-31', tz=tz), Timestamp('2013-02-28', tz=tz), Timestamp('2013-03-31', tz=tz), Timestamp('2013-04-30', tz=tz)]
    expected = Index(expected_list, dtype=object, name='idx')
    result = idx.astype(object)
    tm.assert_index_equal(result, expected)
    assert idx.tolist() == expected_list