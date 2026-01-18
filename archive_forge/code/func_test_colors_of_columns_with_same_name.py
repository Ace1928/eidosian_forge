import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_colors_of_columns_with_same_name(self):
    df = DataFrame({'b': [0, 1, 0], 'a': [1, 2, 3]})
    df1 = DataFrame({'a': [2, 4, 6]})
    df_concat = pd.concat([df, df1], axis=1)
    result = df_concat.plot()
    legend = result.get_legend()
    if Version(mpl.__version__) < Version('3.7'):
        handles = legend.legendHandles
    else:
        handles = legend.legend_handles
    for legend, line in zip(handles, result.lines):
        assert legend.get_color() == line.get_color()