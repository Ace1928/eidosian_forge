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
@pytest.mark.parametrize('indexer', [[0], slice(None, 1, None), np.array([0])])
@pytest.mark.parametrize('value', [['Z'], np.array(['Z'])])
def test_iloc_setitem_with_scalar_index(self, indexer, value):
    df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B']).astype({'A': object})
    df.iloc[0, indexer] = value
    result = df.iloc[0, 0]
    assert is_scalar(result) and result == 'Z'