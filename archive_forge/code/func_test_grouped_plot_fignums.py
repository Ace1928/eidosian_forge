import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_grouped_plot_fignums(self):
    n = 10
    weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
    height = Series(np.random.default_rng(2).normal(60, 10, size=n))
    gender = np.random.default_rng(2).choice(['male', 'female'], size=n)
    df = DataFrame({'height': height, 'weight': weight, 'gender': gender})
    gb = df.groupby('gender')
    res = gb.plot()
    assert len(mpl.pyplot.get_fignums()) == 2
    assert len(res) == 2
    plt.close('all')
    res = gb.boxplot(return_type='axes')
    assert len(mpl.pyplot.get_fignums()) == 1
    assert len(res) == 2