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
def test_constructor_datetime64(self):
    rng = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
    dates = np.asarray(rng)
    series = Series(dates)
    assert np.issubdtype(series.dtype, np.dtype('M8[ns]'))