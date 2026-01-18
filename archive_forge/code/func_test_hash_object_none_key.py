import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_hash_object_none_key():
    result = pd.util.hash_pandas_object(Series(['a', 'b']), hash_key=None)
    expected = Series([4578374827886788867, 17338122309987883691], dtype='uint64')
    tm.assert_series_equal(result, expected)