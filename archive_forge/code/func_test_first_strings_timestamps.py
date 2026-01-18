import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_strings_timestamps():
    test = DataFrame({Timestamp('2012-01-01 00:00:00'): ['a', 'b'], Timestamp('2012-01-02 00:00:00'): ['c', 'd'], 'name': ['e', 'e'], 'aaaa': ['f', 'g']})
    result = test.groupby('name').first()
    expected = DataFrame([['a', 'c', 'f']], columns=Index([Timestamp('2012-01-01'), Timestamp('2012-01-02'), 'aaaa']), index=Index(['e'], name='name'))
    tm.assert_frame_equal(result, expected)