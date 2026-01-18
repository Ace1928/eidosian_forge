import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_df_legend_labels_time_series_no_mutate(self):
    pytest.importorskip('scipy')
    ind = date_range('1/1/2014', periods=3)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=['a', 'b', 'c'], index=ind)
    df5 = df.set_index('a')
    ax = df5.plot(y='b')
    _check_legend_labels(ax, labels=['b'])
    ax = df5.plot(y='b', label='LABEL_b')
    _check_legend_labels(ax, labels=['LABEL_b'])
    _check_text_labels(ax.xaxis.get_label(), 'a')
    ax = df5.plot(y='c', label='LABEL_c', ax=ax)
    _check_legend_labels(ax, labels=['LABEL_b', 'LABEL_c'])
    assert df5.columns.tolist() == ['b', 'c']