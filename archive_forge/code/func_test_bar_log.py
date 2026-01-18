from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
@pytest.mark.parametrize('axis, meth', [('yaxis', 'bar'), ('xaxis', 'barh')])
def test_bar_log(self, axis, meth):
    expected = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    _, ax = mpl.pyplot.subplots()
    ax = getattr(Series([200, 500]).plot, meth)(log=True, ax=ax)
    tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)