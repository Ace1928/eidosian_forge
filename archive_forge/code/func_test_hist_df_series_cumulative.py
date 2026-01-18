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
def test_hist_df_series_cumulative(self):
    from matplotlib.patches import Rectangle
    series = Series(np.random.default_rng(2).random(10))
    ax = series.plot.hist(cumulative=True, bins=4)
    rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]
    tm.assert_almost_equal(rects[-2].get_height(), 10.0)