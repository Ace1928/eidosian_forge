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
def test_dup_datetime_index_plot(self):
    dr1 = date_range('1/1/2009', periods=4)
    dr2 = date_range('1/2/2009', periods=4)
    index = dr1.append(dr2)
    values = np.random.default_rng(2).standard_normal(index.size)
    s = Series(values, index=index)
    _check_plot_works(s.plot)