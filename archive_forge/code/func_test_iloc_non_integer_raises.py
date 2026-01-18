from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('index,columns', [(np.arange(20), list('ABCDE'))])
@pytest.mark.parametrize('index_vals,column_vals', [[slice(None), ['A', 'D']], (['1', '2'], slice(None)), ([datetime(2019, 1, 1)], slice(None))])
def test_iloc_non_integer_raises(self, index, columns, index_vals, column_vals):
    df = DataFrame(np.random.default_rng(2).standard_normal((len(index), len(columns))), index=index, columns=columns)
    msg = '.iloc requires numeric indexers, got'
    with pytest.raises(IndexError, match=msg):
        df.iloc[index_vals, column_vals]