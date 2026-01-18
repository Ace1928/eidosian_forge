from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_str_freq_and_name(self):
    dti = date_range('1/1/2011', periods=3, freq='h', name='test_name')
    result = dti.astype(str)
    expected = Index(['2011-01-01 00:00:00', '2011-01-01 01:00:00', '2011-01-01 02:00:00'], name='test_name', dtype=object)
    tm.assert_index_equal(result, expected)