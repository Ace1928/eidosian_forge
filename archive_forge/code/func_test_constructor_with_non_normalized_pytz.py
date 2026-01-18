from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('tz', ['US/Pacific', 'US/Eastern', 'Asia/Tokyo'])
def test_constructor_with_non_normalized_pytz(self, tz):
    non_norm_tz = Timestamp('2010', tz=tz).tz
    result = DatetimeIndex(['2010'], tz=non_norm_tz)
    assert pytz.timezone(tz) is result.tz