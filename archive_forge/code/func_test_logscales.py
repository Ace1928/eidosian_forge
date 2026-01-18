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
@pytest.mark.parametrize('input_log, expected_log', [(True, 'log'), ('sym', 'symlog')])
def test_logscales(self, input_log, expected_log):
    df = DataFrame({'a': np.arange(100)}, index=np.arange(100))
    ax = df.plot(logy=input_log)
    _check_ax_scales(ax, yaxis=expected_log)
    assert ax.get_yscale() == expected_log
    ax = df.plot(logx=input_log)
    _check_ax_scales(ax, xaxis=expected_log)
    assert ax.get_xscale() == expected_log
    ax = df.plot(loglog=input_log)
    _check_ax_scales(ax, xaxis=expected_log, yaxis=expected_log)
    assert ax.get_xscale() == expected_log
    assert ax.get_yscale() == expected_log