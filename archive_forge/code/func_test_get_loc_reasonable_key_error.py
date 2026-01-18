from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_loc_reasonable_key_error(self):
    index = DatetimeIndex(['1/3/2000'])
    with pytest.raises(KeyError, match='2000'):
        index.get_loc('1/1/2000')