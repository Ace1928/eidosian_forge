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
def test_margins_no_values_one_row_one_col(self, data):
    result = data[['A', 'B']].pivot_table(index='A', columns='B', aggfunc=len, margins=True)
    assert result.All.tolist() == [4.0, 7.0, 11.0]