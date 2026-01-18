from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import delta_to_tick
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_NanosecondGeneric():
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert timestamp.nanosecond == 0
    result = timestamp + Nano(10)
    assert result.nanosecond == 10
    reverse_result = Nano(10) + timestamp
    assert reverse_result.nanosecond == 10