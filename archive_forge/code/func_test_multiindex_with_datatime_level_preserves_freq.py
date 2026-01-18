import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_multiindex_with_datatime_level_preserves_freq(self):
    idx = Index(range(2), name='A')
    dti = pd.date_range('2020-01-01', periods=7, freq='D', name='B')
    mi = MultiIndex.from_product([idx, dti])
    df = DataFrame(np.random.default_rng(2).standard_normal((14, 2)), index=mi)
    result = df.loc[0].index
    tm.assert_index_equal(result, dti)
    assert result.freq == dti.freq