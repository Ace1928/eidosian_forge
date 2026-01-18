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
def test_Hour():
    assert_offset_equal(Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 1))
    assert_offset_equal(Hour(-1), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 2))
    assert_offset_equal(-1 * Hour(), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))
    assert Hour(3) + Hour(2) == Hour(5)
    assert Hour(3) - Hour(2) == Hour()
    assert Hour(4) != Hour(1)