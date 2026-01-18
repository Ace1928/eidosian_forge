import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
def test_mpl2_color_cycle_str(self, color):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
    _check_plot_works(df.plot, color=color)