import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_andrews_curves_handle(self):
    from pandas.plotting import andrews_curves
    colors = ['b', 'g', 'r']
    df = DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [1, 2, 3], 'Name': colors})
    ax = andrews_curves(df, 'Name', color=colors)
    handles, _ = ax.get_legend_handles_labels()
    _check_colors(handles, linecolors=colors)