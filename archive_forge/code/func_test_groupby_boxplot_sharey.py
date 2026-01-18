import pytest
from pandas import DataFrame
from pandas.tests.plotting.common import _check_visible
@pytest.mark.parametrize('kwargs, expected', [({}, [True, False, True, False]), ({'sharey': True}, [True, False, True, False]), ({'sharey': False}, [True, True, True, True])])
def test_groupby_boxplot_sharey(self, kwargs, expected):
    df = DataFrame({'a': [-1.43, -0.15, -3.7, -1.43, -0.14], 'b': [0.56, 0.84, 0.29, 0.56, 0.85], 'c': [0, 1, 2, 3, 1]}, index=[0, 1, 2, 3, 4])
    axes = df.groupby('c').boxplot(**kwargs)
    self._assert_ytickslabels_visibility(axes, expected)