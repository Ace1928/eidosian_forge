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
def test_dont_modify_rcParams(self):
    key = 'axes.prop_cycle'
    colors = mpl.pyplot.rcParams[key]
    _, ax = mpl.pyplot.subplots()
    Series([1, 2, 3]).plot(ax=ax)
    assert colors == mpl.pyplot.rcParams[key]