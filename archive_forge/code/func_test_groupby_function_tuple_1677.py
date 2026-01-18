import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_function_tuple_1677(self):
    df = DataFrame(np.random.default_rng(2).random(100), index=date_range('1/1/2000', periods=100))
    monthly_group = df.groupby(lambda x: (x.year, x.month))
    result = monthly_group.mean()
    assert isinstance(result.index[0], tuple)