import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', object])
def test_empty_method_full_series(self, dtype):
    full_series = Series(index=[1], dtype=dtype)
    assert not full_series.empty