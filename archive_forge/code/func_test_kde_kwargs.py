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
@pytest.mark.parametrize('bw_method, ind', [['scott', 20], [None, 20], [None, np.int_(20)], [0.5, np.linspace(-100, 100, 20)]])
def test_kde_kwargs(self, ts, bw_method, ind):
    pytest.importorskip('scipy')
    _check_plot_works(ts.plot.kde, bw_method=bw_method, ind=ind)