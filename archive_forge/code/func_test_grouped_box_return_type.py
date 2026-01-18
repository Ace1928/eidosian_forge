import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_return_type(self, hist_df):
    df = hist_df
    result = df.boxplot(by='gender')
    assert isinstance(result, np.ndarray)
    _check_box_return_type(result, None, expected_keys=['height', 'weight', 'category'])