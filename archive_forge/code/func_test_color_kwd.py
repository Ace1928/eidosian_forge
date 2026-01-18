import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('colors_kwd, expected', [({'boxes': 'r', 'whiskers': 'b', 'medians': 'g', 'caps': 'c'}, {'boxes': 'r', 'whiskers': 'b', 'medians': 'g', 'caps': 'c'}), ({'boxes': 'r'}, {'boxes': 'r'}), ('r', {'boxes': 'r', 'whiskers': 'r', 'medians': 'r', 'caps': 'r'})])
def test_color_kwd(self, colors_kwd, expected):
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    result = df.boxplot(color=colors_kwd, return_type='dict')
    for k, v in expected.items():
        assert result[k][0].get_color() == v