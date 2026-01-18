import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('method', ['pad', 'backfill', 'nearest'])
def test_get_indexer_with_method_numeric_vs_bool(self, method):
    left = Index([1, 2, 3])
    right = Index([True, False])
    with pytest.raises(TypeError, match='Cannot compare'):
        left.get_indexer(right, method=method)
    with pytest.raises(TypeError, match='Cannot compare'):
        right.get_indexer(left, method=method)