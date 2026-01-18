from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_mismatched_freq(self):
    dti = date_range('2016-01-01', periods=3)
    pi = dti.to_period('D')
    pi2 = dti.to_period('W')
    pi3 = pi.view(pi2.dtype)
    with pytest.raises(KeyError, match='W-SUN'):
        pi.get_loc(pi2[0])
    with pytest.raises(KeyError, match='W-SUN'):
        pi.get_loc(pi3[0])