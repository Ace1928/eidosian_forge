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
@pytest.mark.parametrize('cls', tick_classes)
def test_compare_ticks_to_timedeltalike(cls):
    off = cls(19)
    td = off._as_pd_timedelta
    others = [td, td.to_timedelta64()]
    if cls is not Nano:
        others.append(td.to_pytimedelta())
    for other in others:
        assert off == other
        assert not off != other
        assert not off < other
        assert not off > other
        assert off <= other
        assert off >= other