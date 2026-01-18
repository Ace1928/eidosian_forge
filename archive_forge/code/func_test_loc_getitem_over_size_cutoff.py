from datetime import (
import re
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_over_size_cutoff(monkeypatch):
    monkeypatch.setattr(libindex, '_SIZE_CUTOFF', 1000)
    dates = []
    sec = timedelta(seconds=1)
    half_sec = timedelta(microseconds=500000)
    d = datetime(2011, 12, 5, 20, 30)
    n = 1100
    for i in range(n):
        dates.append(d)
        dates.append(d + sec)
        dates.append(d + sec + half_sec)
        dates.append(d + sec + sec + half_sec)
        d += 3 * sec
    duplicate_positions = np.random.default_rng(2).integers(0, len(dates) - 1, 20)
    for p in duplicate_positions:
        dates[p + 1] = dates[p]
    df = DataFrame(np.random.default_rng(2).standard_normal((len(dates), 4)), index=dates, columns=list('ABCD'))
    pos = n * 3
    timestamp = df.index[pos]
    assert timestamp in df.index
    df.loc[timestamp]
    assert len(df.loc[[timestamp]]) > 0