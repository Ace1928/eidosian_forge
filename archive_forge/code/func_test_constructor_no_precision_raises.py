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
def test_constructor_no_precision_raises(self):
    msg = 'with no precision is not allowed'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(['2000'], dtype='datetime64')
    msg = "The 'datetime64' dtype has no unit. Please pass in"
    with pytest.raises(ValueError, match=msg):
        Index(['2000'], dtype='datetime64')