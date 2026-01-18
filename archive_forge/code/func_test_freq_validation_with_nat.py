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
def test_freq_validation_with_nat(self):
    msg = 'Inferred frequency None from passed values does not conform to passed frequency D'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex([pd.NaT, Timestamp('2011-01-01')], freq='D')
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex([pd.NaT, Timestamp('2011-01-01')._value], freq='D')