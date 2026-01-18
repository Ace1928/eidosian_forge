import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_boxplot_multiindex_column(self):
    arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    tuples = list(zip(*arrays))
    index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 8)), index=['A', 'B', 'C'], columns=index)
    col = [('bar', 'one'), ('bar', 'two')]
    axes = _check_plot_works(df.boxplot, column=col, return_type='axes')
    expected_xticklabel = ['(bar, one)', '(bar, two)']
    result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
    assert expected_xticklabel == result_xticklabel