from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_value_or_method(self, datetime_series):
    msg = "Cannot specify both 'value' and 'method'"
    with pytest.raises(ValueError, match=msg):
        datetime_series.fillna(value=0, method='ffill')