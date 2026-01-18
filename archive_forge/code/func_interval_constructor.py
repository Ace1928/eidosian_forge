import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
@pytest.fixture(params=[IntervalArray.from_arrays, IntervalIndex.from_arrays, create_categorical_intervals, create_series_intervals, create_series_categorical_intervals], ids=['IntervalArray', 'IntervalIndex', 'Categorical[Interval]', 'Series[Interval]', 'Series[Categorical[Interval]]'])
def interval_constructor(self, request):
    """
        Fixture for all pandas native interval constructors.
        To be used as the LHS of IntervalArray comparisons.
        """
    return request.param