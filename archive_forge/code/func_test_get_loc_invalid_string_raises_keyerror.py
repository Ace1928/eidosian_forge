from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_invalid_string_raises_keyerror(self):
    pi = period_range('2000', periods=3, name='A')
    with pytest.raises(KeyError, match='A'):
        pi.get_loc('A')
    ser = Series([1, 2, 3], index=pi)
    with pytest.raises(KeyError, match='A'):
        ser.loc['A']
    with pytest.raises(KeyError, match='A'):
        ser['A']
    assert 'A' not in ser
    assert 'A' not in pi