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
def test_Millisecond():
    assert_offset_equal(Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1000))
    assert_offset_equal(Milli(-1), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1))
    assert_offset_equal(Milli(2), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000))
    assert_offset_equal(2 * Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000))
    assert_offset_equal(-1 * Milli(), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1))
    assert Milli(3) + Milli(2) == Milli(5)
    assert Milli(3) - Milli(2) == Milli()