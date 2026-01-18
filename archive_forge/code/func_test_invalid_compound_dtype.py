from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_invalid_compound_dtype(self):
    c_dtype = np.dtype([('a', 'i8'), ('b', 'f4')])
    cdt_arr = np.array([(1, 0.4), (256, -13)], dtype=c_dtype)
    with pytest.raises(ValueError, match='Use DataFrame instead'):
        Series(cdt_arr, index=['A', 'B'])