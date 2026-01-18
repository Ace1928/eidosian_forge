import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def bind_cols(df):
    iord = lambda a: 0 if a != a else ord(a)
    f = lambda ts: ts.map(iord) - ord('a')
    return f(df['1st']) + f(df['3rd']) * 100.0 + df['2nd'].fillna(0) * 10