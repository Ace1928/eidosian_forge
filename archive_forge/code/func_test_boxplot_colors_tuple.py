import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_boxplot_colors_tuple(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    bp = df.plot.box(color=(0, 1, 0), sym='#123456', return_type='dict')
    _check_colors_box(bp, (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), '#123456')