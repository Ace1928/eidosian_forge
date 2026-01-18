import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_copy_consolidates(self):
    df = DataFrame({'a': np.random.default_rng(2).integers(0, 100, size=55), 'b': np.random.default_rng(2).integers(0, 100, size=55)})
    for i in range(10):
        df.loc[:, f'n_{i}'] = np.random.default_rng(2).integers(0, 100, size=55)
    assert len(df._mgr.blocks) == 11
    result = df.copy()
    assert len(result._mgr.blocks) == 1