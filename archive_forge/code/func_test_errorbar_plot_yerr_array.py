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
@pytest.mark.slow
def test_errorbar_plot_yerr_array(self):
    d = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
    df = DataFrame(d)
    ax = _check_plot_works(df['y'].plot, yerr=np.ones(12) * 0.4)
    _check_has_errorbars(ax, xerr=0, yerr=1)
    ax = _check_plot_works(df.plot, yerr=np.ones((2, 12)) * 0.4)
    _check_has_errorbars(ax, xerr=0, yerr=2)