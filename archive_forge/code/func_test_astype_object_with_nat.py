from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_object_with_nat(self):
    idx = DatetimeIndex([datetime(2013, 1, 1), datetime(2013, 1, 2), NaT, datetime(2013, 1, 4)], name='idx')
    expected_list = [Timestamp('2013-01-01'), Timestamp('2013-01-02'), NaT, Timestamp('2013-01-04')]
    expected = Index(expected_list, dtype=object, name='idx')
    result = idx.astype(object)
    tm.assert_index_equal(result, expected)
    assert idx.tolist() == expected_list