import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_hash_pandas_series_diff_index(series):
    a = hash_pandas_object(series, index=True)
    b = hash_pandas_object(series, index=False)
    assert not (a == b).all()