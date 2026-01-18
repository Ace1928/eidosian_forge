from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('expected', [DataFrame(), DataFrame(columns=list('ABC')), DataFrame(index=list('ABC')), DataFrame({'A': [], 'B': [], 'C': []})])
@pytest.mark.parametrize('func', [round, lambda x: x])
def test_map_empty(expected, func):
    result = expected.map(func)
    tm.assert_frame_equal(result, expected)