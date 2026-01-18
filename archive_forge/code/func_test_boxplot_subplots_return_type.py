import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('rt', ['dict', 'axes', 'both'])
def test_boxplot_subplots_return_type(self, hist_df, rt):
    df = hist_df
    returned = df.plot.box(return_type=rt, subplots=True)
    _check_box_return_type(returned, rt, expected_keys=['height', 'weight', 'category'], check_ax_title=False)