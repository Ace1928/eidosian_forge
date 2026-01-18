import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('scheme,expected', [('dark_background', {'boxes': '#8dd3c7', 'whiskers': '#8dd3c7', 'medians': '#bfbbd9', 'caps': '#8dd3c7'}), ('default', {'boxes': '#1f77b4', 'whiskers': '#1f77b4', 'medians': '#2ca02c', 'caps': '#1f77b4'})])
def test_colors_in_theme(self, scheme, expected):
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    import matplotlib.pyplot as plt
    plt.style.use(scheme)
    result = df.plot.box(return_type='dict')
    for k, v in expected.items():
        assert result[k][0].get_color() == v