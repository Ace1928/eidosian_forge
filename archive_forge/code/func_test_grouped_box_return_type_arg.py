import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
def test_grouped_box_return_type_arg(self, hist_df, return_type):
    df = hist_df
    returned = df.groupby('classroom').boxplot(return_type=return_type)
    _check_box_return_type(returned, return_type, expected_keys=['A', 'B', 'C'])
    returned = df.boxplot(by='classroom', return_type=return_type)
    _check_box_return_type(returned, return_type, expected_keys=['height', 'weight', 'category'])