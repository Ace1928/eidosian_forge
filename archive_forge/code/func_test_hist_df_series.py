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
def test_hist_df_series(self):
    series = Series(np.random.default_rng(2).random(10))
    axes = series.plot.hist(rot=40)
    _check_ticks_props(axes, xrot=40, yrot=0)