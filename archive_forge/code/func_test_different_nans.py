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
def test_different_nans(self):
    comps = [float('nan')]
    values = [float('nan')]
    assert comps[0] is not values[0]
    result = algos.isin(np.array(comps), values)
    tm.assert_numpy_array_equal(np.array([True]), result)
    result = algos.isin(np.asarray(comps, dtype=object), np.asarray(values, dtype=object))
    tm.assert_numpy_array_equal(np.array([True]), result)
    result = algos.isin(np.asarray(comps, dtype=np.float64), np.asarray(values, dtype=np.float64))
    tm.assert_numpy_array_equal(np.array([True]), result)