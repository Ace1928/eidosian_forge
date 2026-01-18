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
def test_different_nan_objects(self):
    comps = np.array(['nan', np.nan * 1j, float('nan')], dtype=object)
    vals = np.array([float('nan')], dtype=object)
    expected = np.array([False, False, True])
    result = algos.isin(comps, vals)
    tm.assert_numpy_array_equal(expected, result)