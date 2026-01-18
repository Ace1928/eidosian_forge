import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_alternate_encoding(index):
    obj = Series(list('abc'))
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    tm.assert_series_equal(a, b)