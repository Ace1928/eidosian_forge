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
def test_bar_nan_stacked(self):
    df = DataFrame({'A': [10, np.nan, 20], 'B': [5, 10, 20], 'C': [1, 2, 3]})
    ax = df.plot.bar(stacked=True)
    expected = [10, 0, 20, 5, 10, 20, 1, 2, 3]
    result = [p.get_height() for p in ax.patches]
    assert result == expected
    result = [p.get_y() for p in ax.patches]
    expected = [0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 15.0, 10.0, 40.0]
    assert result == expected