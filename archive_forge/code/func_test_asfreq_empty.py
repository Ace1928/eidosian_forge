from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_empty(self, datetime_frame):
    zero_length = datetime_frame.reindex([])
    result = zero_length.asfreq('BME')
    assert result is not zero_length