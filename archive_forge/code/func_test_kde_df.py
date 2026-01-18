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
def test_kde_df(self):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).standard_normal((100, 4)))
    ax = _check_plot_works(df.plot, kind='kde')
    expected = [pprint_thing(c) for c in df.columns]
    _check_legend_labels(ax, labels=expected)
    _check_ticks_props(ax, xrot=0)