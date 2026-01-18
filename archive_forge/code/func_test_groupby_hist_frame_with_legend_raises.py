import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('column', [None, 'b'])
def test_groupby_hist_frame_with_legend_raises(self, column):
    index = Index(15 * ['1'] + 15 * ['2'], name='c')
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 2)), index=index, columns=['a', 'b'])
    g = df.groupby('c')
    with pytest.raises(ValueError, match='Cannot use both legend and label'):
        g.hist(legend=True, column=column, label='d')