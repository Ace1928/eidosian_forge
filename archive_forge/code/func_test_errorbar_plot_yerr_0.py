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
@pytest.mark.slow
def test_errorbar_plot_yerr_0(self):
    s = Series(np.arange(10), name='x')
    s_err = np.abs(np.random.default_rng(2).standard_normal(10))
    ax = _check_plot_works(s.plot, xerr=s_err)
    _check_has_errorbars(ax, xerr=1, yerr=0)