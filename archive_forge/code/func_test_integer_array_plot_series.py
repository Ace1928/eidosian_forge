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
@pytest.mark.parametrize('plot', ['line', 'bar', 'hist', 'pie'])
def test_integer_array_plot_series(self, plot):
    arr = pd.array([1, 2, 3, 4], dtype='UInt32')
    s = Series(arr)
    _check_plot_works(getattr(s.plot, plot))