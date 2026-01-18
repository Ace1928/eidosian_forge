from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.filterwarnings('ignore:Attempt:UserWarning')
def test_boxplot_vertical_subplots(self, hist_df):
    df = hist_df
    numeric_cols = df._get_numeric_data().columns
    labels = [pprint_thing(c) for c in numeric_cols]
    axes = _check_plot_works(df.plot.box, default_axes=True, subplots=True, vert=False, logx=True)
    _check_axes_shape(axes, axes_num=3, layout=(1, 3))
    _check_ax_scales(axes, xaxis='log')
    for ax, label in zip(axes, labels):
        _check_text_labels(ax.get_yticklabels(), [label])
        assert len(ax.lines) == 7