import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['cov', 'corr'])
def test_corr_cov_independent_index_column(self, method):
    df = DataFrame(np.random.default_rng(2).standard_normal(4 * 10).reshape(10, 4), columns=list('abcd'))
    result = getattr(df, method)()
    assert result.index is not result.columns
    assert result.index.equals(result.columns)