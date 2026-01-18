import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_asof_periodindex_mismatched_freq(self):
    N = 50
    rng = period_range('1/1/1990', periods=N, freq='h')
    df = DataFrame(np.random.default_rng(2).standard_normal(N), index=rng)
    msg = 'Input has different freq'
    with pytest.raises(IncompatibleFrequency, match=msg):
        df.asof(rng.asfreq('D'))