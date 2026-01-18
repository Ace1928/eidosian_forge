import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_filter_corner(self):
    empty = DataFrame()
    result = empty.filter([])
    tm.assert_frame_equal(result, empty)
    result = empty.filter(like='foo')
    tm.assert_frame_equal(result, empty)