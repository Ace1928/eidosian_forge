from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_non_unique_index_many_to_many(self):
    dt = datetime(2012, 5, 1)
    dt2 = datetime(2012, 5, 2)
    dt3 = datetime(2012, 5, 3)
    df1 = DataFrame({'x': ['a', 'b', 'c', 'd']}, index=[dt2, dt2, dt, dt])
    df2 = DataFrame({'y': ['e', 'f', 'g', ' h', 'i']}, index=[dt2, dt2, dt3, dt, dt])
    _check_merge(df1, df2)