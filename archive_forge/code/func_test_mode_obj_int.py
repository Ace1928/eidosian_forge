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
def test_mode_obj_int(self):
    exp = Series([1], dtype=int)
    tm.assert_numpy_array_equal(algos.mode(exp.values), exp.values)
    exp = Series(['a', 'b', 'c'], dtype=object)
    tm.assert_numpy_array_equal(algos.mode(exp.values), exp.values)