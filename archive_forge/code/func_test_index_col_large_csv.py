from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.slow
def test_index_col_large_csv(all_parsers, monkeypatch):
    parser = all_parsers
    ARR_LEN = 100
    df = DataFrame({'a': range(ARR_LEN + 1), 'b': np.random.default_rng(2).standard_normal(ARR_LEN + 1)})
    with tm.ensure_clean() as path:
        df.to_csv(path, index=False)
        with monkeypatch.context() as m:
            m.setattr('pandas.core.algorithms._MINIMUM_COMP_ARR_LEN', ARR_LEN)
            result = parser.read_csv(path, index_col=[0])
    tm.assert_frame_equal(result, df.set_index('a'))