from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def test_frame_operators(self, float_frame):
    frame = float_frame
    garbage = np.random.default_rng(2).random(4)
    colSeries = Series(garbage, index=np.array(frame.columns))
    idSum = frame + frame
    seriesSum = frame + colSeries
    for col, series in idSum.items():
        for idx, val in series.items():
            origVal = frame[col][idx] * 2
            if not np.isnan(val):
                assert val == origVal
            else:
                assert np.isnan(origVal)
    for col, series in seriesSum.items():
        for idx, val in series.items():
            origVal = frame[col][idx] + colSeries[col]
            if not np.isnan(val):
                assert val == origVal
            else:
                assert np.isnan(origVal)