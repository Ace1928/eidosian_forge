from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_tz_conversion_freq(self, tz_naive_fixture):
    t3 = DatetimeIndex(['2019-01-01 10:00'], freq='h')
    assert t3.tz_localize(tz=tz_naive_fixture).freq == t3.freq
    t4 = DatetimeIndex(['2019-01-02 12:00'], tz='UTC', freq='min')
    assert t4.tz_convert(tz='UTC').freq == t4.freq