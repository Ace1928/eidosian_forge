import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_boxplot_legacy2_with_ax_return_type(self):
    df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=['Col1', 'Col2'])
    df['X'] = Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
    df['Y'] = Series(['A'] * 10)
    fig, ax = mpl.pyplot.subplots()
    axes = df.groupby('Y').boxplot(ax=ax, return_type='axes')
    ax_axes = ax.axes
    assert ax_axes is axes['A']