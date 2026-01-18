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
@pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
def test_invalid_plot_data(self, kind):
    s = Series(list('abcd'))
    _, ax = mpl.pyplot.subplots()
    msg = 'no numeric data to plot'
    with pytest.raises(TypeError, match=msg):
        s.plot(kind=kind, ax=ax)