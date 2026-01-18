import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_return_type_groupby(self, hist_df):
    df = hist_df
    result = df.groupby('gender').boxplot(return_type='dict')
    _check_box_return_type(result, 'dict', expected_keys=['Male', 'Female'])