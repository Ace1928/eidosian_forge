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
def test_datelike_mode(self):
    exp = Series(['1900-05-03', '2011-01-03', '2013-01-02'], dtype='M8[ns]')
    ser = Series(['2011-01-03', '2013-01-02', '1900-05-03'], dtype='M8[ns]')
    tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
    tm.assert_series_equal(ser.mode(), exp)
    exp = Series(['2011-01-03', '2013-01-02'], dtype='M8[ns]')
    ser = Series(['2011-01-03', '2013-01-02', '1900-05-03', '2011-01-03', '2013-01-02'], dtype='M8[ns]')
    tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
    tm.assert_series_equal(ser.mode(), exp)