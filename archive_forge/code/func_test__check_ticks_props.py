import pytest
from pandas import DataFrame
from pandas.tests.plotting.common import (
def test__check_ticks_props(self):
    df = DataFrame({'b': [0, 1, 0], 'a': [1, 2, 3]})
    ax = _check_plot_works(df.plot, rot=30)
    ax.yaxis.set_tick_params(rotation=30)
    msg = 'expected 0.00000 but got '
    with pytest.raises(AssertionError, match=msg):
        _check_ticks_props(ax, xrot=0)
    with pytest.raises(AssertionError, match=msg):
        _check_ticks_props(ax, xlabelsize=0)
    with pytest.raises(AssertionError, match=msg):
        _check_ticks_props(ax, yrot=0)
    with pytest.raises(AssertionError, match=msg):
        _check_ticks_props(ax, ylabelsize=0)