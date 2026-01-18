import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_line_colors_single_color(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot(color='red')
    _check_colors(ax.get_lines(), linecolors=['red'] * 5)