import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_hist_df_legacy_external_error(self):
    ser = Series(range(10))
    with tm.external_error_raised(AttributeError):
        ser.hist(foo='bar')