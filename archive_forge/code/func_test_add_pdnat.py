from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_add_pdnat(self, tda):
    result = tda + pd.NaT
    assert isinstance(result, TimedeltaArray)
    assert result._creso == tda._creso
    assert result.isna().all()
    result = pd.NaT + tda
    assert isinstance(result, TimedeltaArray)
    assert result._creso == tda._creso
    assert result.isna().all()