from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_infer_from_tdi(self):
    tdi = timedelta_range('1 second', periods=10 ** 7, freq='1s')
    result = TimedeltaIndex(tdi, freq='infer')
    assert result.freq == tdi.freq
    assert 'inferred_freq' not in getattr(result, '_cache', {})