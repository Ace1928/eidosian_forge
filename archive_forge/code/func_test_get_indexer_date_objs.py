from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_indexer_date_objs(self):
    rng = date_range('1/1/2000', periods=20)
    result = rng.get_indexer(rng.map(lambda x: x.date()))
    expected = rng.get_indexer(rng)
    tm.assert_numpy_array_equal(result, expected)