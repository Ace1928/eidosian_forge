from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_ground_truth(self, samples, actual_kurt):
    kurt = nanops.nankurt(samples)
    tm.assert_almost_equal(kurt, actual_kurt)