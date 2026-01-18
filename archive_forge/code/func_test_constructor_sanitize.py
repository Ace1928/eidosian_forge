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
def test_constructor_sanitize(self):
    s = Series(np.array([1.0, 1.0, 8.0]), dtype='i8')
    assert s.dtype == np.dtype('i8')
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(np.array([1.0, 1.0, np.nan]), copy=True, dtype='i8')