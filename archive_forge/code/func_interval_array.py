import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
@pytest.fixture
def interval_array(left_right_dtypes):
    """
    Fixture to generate an IntervalArray of various dtypes containing NA if possible
    """
    left, right = left_right_dtypes
    return IntervalArray.from_arrays(left, right)