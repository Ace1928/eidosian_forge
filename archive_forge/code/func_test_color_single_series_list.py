import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_color_single_series_list(self):
    df = DataFrame({'A': [1, 2, 3]})
    _check_plot_works(df.plot, color=['red'])