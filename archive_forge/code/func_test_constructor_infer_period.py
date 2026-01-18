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
@pytest.mark.parametrize('data_constructor', [list, np.array], ids=['list', 'ndarray[object]'])
def test_constructor_infer_period(self, data_constructor):
    data = [Period('2000', 'D'), Period('2001', 'D'), None]
    result = Series(data_constructor(data))
    expected = Series(period_array(data))
    tm.assert_series_equal(result, expected)
    assert result.dtype == 'Period[D]'