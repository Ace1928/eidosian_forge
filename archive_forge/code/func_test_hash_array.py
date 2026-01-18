import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_hash_array(series):
    arr = series.values
    tm.assert_numpy_array_equal(hash_array(arr), hash_array(arr))