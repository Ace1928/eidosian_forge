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
@pytest.mark.parametrize('err_box', [lambda x: x, DataFrame])
def test_errorbar_with_partial_columns_box(self, err_box):
    d = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
    df = DataFrame(d)
    err = err_box({'x': np.ones(12) * 0.2, 'z': np.ones(12) * 0.4})
    ax = _check_plot_works(df.plot, yerr=err)
    _check_has_errorbars(ax, xerr=0, yerr=1)