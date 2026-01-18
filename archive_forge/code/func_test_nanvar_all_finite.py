from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nanvar_all_finite(self, samples, variance):
    actual_variance = nanops.nanvar(samples)
    tm.assert_almost_equal(actual_variance, variance, rtol=0.01)