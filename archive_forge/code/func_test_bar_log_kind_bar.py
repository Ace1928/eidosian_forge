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
@pytest.mark.parametrize('axis, kind, res_meth', [['yaxis', 'bar', 'get_ylim'], ['xaxis', 'barh', 'get_xlim']])
def test_bar_log_kind_bar(self, axis, kind, res_meth):
    expected = np.array([1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
    _, ax = mpl.pyplot.subplots()
    ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind=kind, ax=ax)
    ymin = 0.0007943282347242822
    ymax = 0.12589254117941673
    res = getattr(ax, res_meth)()
    tm.assert_almost_equal(res[0], ymin)
    tm.assert_almost_equal(res[1], ymax)
    tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)