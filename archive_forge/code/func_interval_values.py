from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.fixture(params=[([0] * 4, [1] * 4), (range(3), range(1, 4))])
def interval_values(request, closed):
    left, right = request.param
    return Categorical(pd.IntervalIndex.from_arrays(left, right, closed))