from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_errorbar_scatter(self):
    df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((5, 2))), index=range(5), columns=['x', 'y'])
    df_err = DataFrame(np.abs(np.random.default_rng(2).standard_normal((5, 2))) / 5, index=range(5), columns=['x', 'y'])
    ax = _check_plot_works(df.plot.scatter, x='x', y='y')
    _check_has_errorbars(ax, xerr=0, yerr=0)
    ax = _check_plot_works(df.plot.scatter, x='x', y='y', xerr=df_err)
    _check_has_errorbars(ax, xerr=1, yerr=0)
    ax = _check_plot_works(df.plot.scatter, x='x', y='y', yerr=df_err)
    _check_has_errorbars(ax, xerr=0, yerr=1)
    ax = _check_plot_works(df.plot.scatter, x='x', y='y', xerr=df_err, yerr=df_err)
    _check_has_errorbars(ax, xerr=1, yerr=1)