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
def test_categorical_zeroes(self):
    s = Series(Categorical(list('bbbaac'), categories=list('abcd'), ordered=True))
    result = s.value_counts()
    expected = Series([3, 2, 1, 0], index=Categorical(['b', 'a', 'c', 'd'], categories=list('abcd'), ordered=True), name='count')
    tm.assert_series_equal(result, expected, check_index_type=True)