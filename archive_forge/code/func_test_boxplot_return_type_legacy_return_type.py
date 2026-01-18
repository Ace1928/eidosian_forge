import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
def test_boxplot_return_type_legacy_return_type(self, return_type):
    df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
    with tm.assert_produces_warning(False):
        result = df.boxplot(return_type=return_type)
    _check_box_return_type(result, return_type)