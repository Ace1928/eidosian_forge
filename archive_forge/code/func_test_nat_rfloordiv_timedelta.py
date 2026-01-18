from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('val,expected', [(np.nan, NaT), (NaT, np.nan), (np.timedelta64('NaT'), np.nan)])
def test_nat_rfloordiv_timedelta(val, expected):
    td = Timedelta(hours=3, minutes=4)
    assert td // val is expected