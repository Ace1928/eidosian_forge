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
@pytest.mark.parametrize('prefix', ['', 'dateutil/'])
def test_dti_constructor_static_tzinfo(self, prefix):
    index = DatetimeIndex([datetime(2012, 1, 1)], tz=prefix + 'EST')
    index.hour
    index[0]