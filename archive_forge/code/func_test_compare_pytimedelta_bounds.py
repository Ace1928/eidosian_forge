from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.skip_ubsan
def test_compare_pytimedelta_bounds(self):
    for unit in ['ns', 'us']:
        tdmax = Timedelta.max.as_unit(unit).max
        tdmin = Timedelta.min.as_unit(unit).min
        assert tdmax < timedelta.max
        assert tdmax <= timedelta.max
        assert not tdmax > timedelta.max
        assert not tdmax >= timedelta.max
        assert tdmax != timedelta.max
        assert not tdmax == timedelta.max
        assert tdmin > timedelta.min
        assert tdmin >= timedelta.min
        assert not tdmin < timedelta.min
        assert not tdmin <= timedelta.min
        assert tdmin != timedelta.min
        assert not tdmin == timedelta.min
    for unit in ['ms', 's']:
        tdmax = Timedelta.max.as_unit(unit).max
        tdmin = Timedelta.min.as_unit(unit).min
        assert tdmax > timedelta.max
        assert tdmax >= timedelta.max
        assert not tdmax < timedelta.max
        assert not tdmax <= timedelta.max
        assert tdmax != timedelta.max
        assert not tdmax == timedelta.max
        assert tdmin < timedelta.min
        assert tdmin <= timedelta.min
        assert not tdmin > timedelta.min
        assert not tdmin >= timedelta.min
        assert tdmin != timedelta.min
        assert not tdmin == timedelta.min