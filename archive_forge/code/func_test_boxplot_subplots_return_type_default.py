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
def test_boxplot_subplots_return_type_default(self, hist_df):
    df = hist_df
    result = df.plot.box(subplots=True)
    assert isinstance(result, Series)
    _check_box_return_type(result, None, expected_keys=['height', 'weight', 'category'])