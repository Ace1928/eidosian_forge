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
def test_bar_bottom_left_left(self):
    df = DataFrame(np.random.default_rng(2).random((5, 5)))
    ax = df.plot.barh(stacked=False, left=np.array([1, 1, 1, 1, 1]))
    result = [p.get_x() for p in ax.patches]
    assert result == [1] * 25
    ax = df.plot.barh(stacked=True, left=[1, 2, 3, 4, 5])
    result = [p.get_x() for p in ax.patches[:5]]
    assert result == [1, 2, 3, 4, 5]