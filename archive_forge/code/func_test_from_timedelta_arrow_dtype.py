from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('unit', ['ns', 'ms'])
def test_from_timedelta_arrow_dtype(unit):
    pytest.importorskip('pyarrow')
    expected = Series([timedelta(1)], dtype=f'duration[{unit}][pyarrow]')
    result = to_timedelta(expected)
    tm.assert_series_equal(result, expected)