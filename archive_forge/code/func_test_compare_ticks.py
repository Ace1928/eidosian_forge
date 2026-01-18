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
def test_compare_ticks(cls):
    three = cls(3)
    four = cls(4)
    assert three < cls(4)
    assert cls(3) < four
    assert four > cls(3)
    assert cls(4) > three
    assert cls(3) == cls(3)
    assert cls(3) != cls(4)