import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_boxplot_colors_dict_colors(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    dict_colors = {'boxes': '#572923', 'whiskers': '#982042', 'medians': '#804823', 'caps': '#123456'}
    bp = df.plot.box(color=dict_colors, sym='r+', return_type='dict')
    _check_colors_box(bp, dict_colors['boxes'], dict_colors['whiskers'], dict_colors['medians'], dict_colors['caps'], 'r')