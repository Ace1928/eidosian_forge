from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_diff_ea_axis(self):
    dta = date_range('2016-01-01', periods=3, tz='US/Pacific')._data
    msg = 'cannot diff DatetimeArray on axis=1'
    with pytest.raises(ValueError, match=msg):
        algos.diff(dta, 1, axis=1)