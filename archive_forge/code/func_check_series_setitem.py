from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
def check_series_setitem(self, elem, index: Index, inplace: bool):
    arr = index._data.copy()
    ser = Series(arr, copy=False)
    self.check_can_hold_element(ser, elem, inplace)
    if is_scalar(elem):
        ser[0] = elem
    else:
        ser[:len(elem)] = elem
    if inplace:
        assert ser.array is arr
    else:
        assert ser.dtype == object