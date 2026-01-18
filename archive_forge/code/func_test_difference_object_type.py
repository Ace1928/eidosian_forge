from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('diff_type, expected', [['difference', [1, 'B']], ['symmetric_difference', [1, 2, 'B', 'C']]])
def test_difference_object_type(self, diff_type, expected):
    idx1 = Index([0, 1, 'A', 'B'])
    idx2 = Index([0, 2, 'A', 'C'])
    result = getattr(idx1, diff_type)(idx2)
    expected = Index(expected)
    tm.assert_index_equal(result, expected)