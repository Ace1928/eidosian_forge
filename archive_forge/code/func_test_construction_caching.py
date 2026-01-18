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
def test_construction_caching(self):
    df = pd.DataFrame({'dt': date_range('20130101', periods=3), 'dttz': date_range('20130101', periods=3, tz='US/Eastern'), 'dt_with_null': [Timestamp('20130101'), pd.NaT, Timestamp('20130103')], 'dtns': date_range('20130101', periods=3, freq='ns')})
    assert df.dttz.dtype.tz.zone == 'US/Eastern'