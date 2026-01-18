import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [None, 'Int64'])
def test_integer_series_size(self, dtype):
    s = Series(range(9), dtype=dtype)
    assert s.size == 9