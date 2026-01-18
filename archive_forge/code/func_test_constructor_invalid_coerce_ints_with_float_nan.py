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
def test_constructor_invalid_coerce_ints_with_float_nan(self, any_int_numpy_dtype):
    vals = [1, 2, np.nan]
    msg = 'cannot convert float NaN to integer'
    with pytest.raises(ValueError, match=msg):
        Series(vals, dtype=any_int_numpy_dtype)
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(np.array(vals), dtype=any_int_numpy_dtype)