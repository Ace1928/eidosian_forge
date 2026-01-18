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
@pytest.mark.xfail(strict=False, reason='2020-12-01 this has been failing periodically on the ymin==0 assertion for a week or so.')
@pytest.mark.parametrize('stacked', [True, False])
def test_area_lim(self, stacked):
    df = DataFrame(np.random.default_rng(2).random((6, 4)), columns=['x', 'y', 'z', 'four'])
    neg_df = -df
    ax = _check_plot_works(df.plot.area, stacked=stacked)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lines = ax.get_lines()
    assert xmin <= lines[0].get_data()[0][0]
    assert xmax >= lines[0].get_data()[0][-1]
    assert ymin == 0
    ax = _check_plot_works(neg_df.plot.area, stacked=stacked)
    ymin, ymax = ax.get_ylim()
    assert ymax == 0