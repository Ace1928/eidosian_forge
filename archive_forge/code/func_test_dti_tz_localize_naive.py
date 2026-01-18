from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_dti_tz_localize_naive(self):
    rng = date_range('1/1/2011', periods=100, freq='h')
    conv = rng.tz_localize('US/Pacific')
    exp = date_range('1/1/2011', periods=100, freq='h', tz='US/Pacific')
    tm.assert_index_equal(conv, exp._with_freq(None))