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
def test_boxplot_vertical_positions(self, hist_df):
    df = hist_df
    numeric_cols = df._get_numeric_data().columns
    labels = [pprint_thing(c) for c in numeric_cols]
    positions = np.array([3, 2, 8])
    ax = df.plot.box(positions=positions, vert=False)
    _check_text_labels(ax.get_yticklabels(), labels)
    tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), positions)
    assert len(ax.lines) == 7 * len(numeric_cols)