import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_invert_array():
    df = pd.DataFrame({'a': pd.date_range('20190101', periods=3, tz='UTC')})
    listify = df.apply(lambda x: x.array, axis=1)
    result = listify.explode()
    tm.assert_series_equal(result, df['a'].rename())