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
def test_density_kwargs(self, ts):
    pytest.importorskip('scipy')
    sample_points = np.linspace(-100, 100, 20)
    _check_plot_works(ts.plot.density, bw_method=0.5, ind=sample_points)