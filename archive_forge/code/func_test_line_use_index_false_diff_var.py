from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_line_use_index_false_diff_var(self):
    s = Series([1, 2, 3], index=['a', 'b', 'c'])
    s.index.name = 'The Index'
    _, ax = mpl.pyplot.subplots()
    ax2 = s.plot.bar(use_index=False, ax=ax)
    label2 = ax2.get_xlabel()
    assert label2 == ''