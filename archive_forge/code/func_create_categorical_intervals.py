import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
def create_categorical_intervals(left, right, closed='right'):
    return Categorical(IntervalIndex.from_arrays(left, right, closed))