import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('op', ['year', 'day', 'second', 'weekday'])
def test_datetime_series_no_datelike_attrs(self, op, datetime_series):
    msg = f"'Series' object has no attribute '{op}'"
    with pytest.raises(AttributeError, match=msg):
        getattr(datetime_series, op)