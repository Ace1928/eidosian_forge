from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_dt64tzindex_mismatched_tzs():
    idx1 = date_range('2001', periods=5, freq='h', tz='US/Eastern')
    ser = Series(np.random.default_rng(2).standard_normal(len(idx1)), index=idx1)
    ser_central = ser.tz_convert('US/Central')
    new1, new2 = ser.align(ser_central)
    assert new1.index.tz is timezone.utc
    assert new2.index.tz is timezone.utc