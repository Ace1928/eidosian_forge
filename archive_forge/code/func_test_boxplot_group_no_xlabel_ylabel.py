import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('vert', [True, False])
def test_boxplot_group_no_xlabel_ylabel(self, vert):
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
    ax = df.boxplot(by='group', vert=vert)
    for subplot in ax:
        target_label = subplot.get_xlabel() if vert else subplot.get_ylabel()
        assert target_label == pprint_thing(['group'])
    mpl.pyplot.close()