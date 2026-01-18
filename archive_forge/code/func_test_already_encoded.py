import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_already_encoded(index):
    obj = Series(list('abc')).str.encode('utf8')
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)