import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_df_subplots_patterns_minorticks(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=date_range('1/1/2000', periods=10), columns=list('AB'))
    _, axes = plt.subplots(2, 1, sharex=True)
    axes = df.plot(subplots=True, ax=axes)
    for ax in axes:
        assert len(ax.lines) == 1
        _check_visible(ax.get_yticklabels(), visible=True)
    _check_visible(axes[0].get_xticklabels(), visible=False)
    _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
    _check_visible(axes[1].get_xticklabels(), visible=True)
    _check_visible(axes[1].get_xticklabels(minor=True), visible=True)