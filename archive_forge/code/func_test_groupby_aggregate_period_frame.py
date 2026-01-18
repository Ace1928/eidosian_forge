import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', ['min', 'max'])
def test_groupby_aggregate_period_frame(func):
    groups = [1, 2]
    periods = pd.period_range('2020', periods=2, freq='Y')
    df = DataFrame({'a': groups, 'b': periods})
    result = getattr(df.groupby('a'), func)()
    idx = Index([1, 2], name='a')
    expected = DataFrame({'b': periods}, index=idx)
    tm.assert_frame_equal(result, expected)