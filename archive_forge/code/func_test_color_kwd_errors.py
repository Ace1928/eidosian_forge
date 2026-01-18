import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('dict_colors, msg', [({'boxes': 'r', 'invalid_key': 'r'}, "invalid key 'invalid_key'")])
def test_color_kwd_errors(self, dict_colors, msg):
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    with pytest.raises(ValueError, match=msg):
        df.boxplot(color=dict_colors, return_type='dict')