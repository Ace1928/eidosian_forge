from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
@pytest.mark.parametrize('attribute', ['hours', 'days', 'weeks', 'months', 'years'])
def test_dateoffset_immutable(attribute):
    offset = DateOffset(**{attribute: 0})
    msg = 'DateOffset objects are immutable'
    with pytest.raises(AttributeError, match=msg):
        setattr(offset, attribute, 5)