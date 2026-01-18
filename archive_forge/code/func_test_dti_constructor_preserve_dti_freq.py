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
def test_dti_constructor_preserve_dti_freq(self):
    rng = date_range('1/1/2000', '1/2/2000', freq='5min')
    rng2 = DatetimeIndex(rng)
    assert rng.freq == rng2.freq