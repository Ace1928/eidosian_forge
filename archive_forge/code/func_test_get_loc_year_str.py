from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_loc_year_str(self):
    rng = date_range('1/1/2000', '1/1/2010')
    result = rng.get_loc('2009')
    expected = slice(3288, 3653)
    assert result == expected