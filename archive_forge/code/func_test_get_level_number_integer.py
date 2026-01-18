import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_level_number_integer(idx):
    idx.names = [1, 0]
    assert idx._get_level_number(1) == 0
    assert idx._get_level_number(0) == 1
    msg = 'Too many levels: Index has only 2 levels, not 3'
    with pytest.raises(IndexError, match=msg):
        idx._get_level_number(2)
    with pytest.raises(KeyError, match='Level fourth not found'):
        idx._get_level_number('fourth')