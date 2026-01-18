import pytest
from pandas import DataFrame
from pandas.tests.plotting.common import (
def test__gen_two_subplots_with_ax(self):
    fig = plt.gcf()
    gen = _gen_two_subplots(f=lambda **kwargs: None, fig=fig, ax='test')
    next(gen)
    assert fig.get_axes() == []
    next(gen)
    axes = fig.get_axes()
    assert len(axes) == 1
    subplot_geometry = list(axes[0].get_subplotspec().get_geometry()[:-1])
    subplot_geometry[-1] += 1
    assert subplot_geometry == [2, 1, 2]