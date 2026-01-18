import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_kde_density_works(self, ts):
    pytest.importorskip('scipy')
    _check_plot_works(ts.plot.density)