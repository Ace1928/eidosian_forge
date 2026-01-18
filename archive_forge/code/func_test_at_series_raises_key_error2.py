from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_series_raises_key_error2(self, indexer_al):
    ser = Series([1, 2, 3], index=list('abc'))
    result = indexer_al(ser)['a']
    assert result == 1
    with pytest.raises(KeyError, match='^0$'):
        indexer_al(ser)[0]