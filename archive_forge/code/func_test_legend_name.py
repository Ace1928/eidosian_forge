import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_legend_name(self):
    multi = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=[np.array(['a', 'a', 'b', 'b']), np.array(['x', 'y', 'x', 'y'])])
    multi.columns.names = ['group', 'individual']
    ax = multi.plot()
    leg_title = ax.legend_.get_title()
    _check_text_labels(leg_title, 'group,individual')
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot(legend=True, ax=ax)
    leg_title = ax.legend_.get_title()
    _check_text_labels(leg_title, 'group,individual')
    df.columns.name = 'new'
    ax = df.plot(legend=False, ax=ax)
    leg_title = ax.legend_.get_title()
    _check_text_labels(leg_title, 'group,individual')
    ax = df.plot(legend=True, ax=ax)
    leg_title = ax.legend_.get_title()
    _check_text_labels(leg_title, 'new')