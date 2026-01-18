from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_str_nat(self):
    idx = DatetimeIndex(['2016-05-16', 'NaT', NaT, np.nan])
    result = idx.astype(str)
    expected = Index(['2016-05-16', 'NaT', 'NaT', 'NaT'], dtype=object)
    tm.assert_index_equal(result, expected)