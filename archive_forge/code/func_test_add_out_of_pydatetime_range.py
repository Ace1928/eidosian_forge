from __future__ import annotations
from datetime import datetime
import numpy as np
import pytest
from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_add_out_of_pydatetime_range():
    ts = Timestamp(np.datetime64('-20000-12-31'))
    off = YearEnd()
    result = ts + off
    assert result.year in (-19999, 1973)
    assert result.month == 12
    assert result.day == 31