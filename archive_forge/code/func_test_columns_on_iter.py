import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_columns_on_iter():
    df = pd.DataFrame({k: range(10) for k in 'ABC'})
    cols = ['A', 'B']
    for _, dg in df.groupby(df.A < 4)[cols]:
        tm.assert_index_equal(dg.columns, pd.Index(cols))
        assert 'C' not in dg.columns