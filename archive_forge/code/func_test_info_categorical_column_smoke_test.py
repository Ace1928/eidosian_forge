from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_info_categorical_column_smoke_test():
    n = 2500
    df = DataFrame({'int64': np.random.randint(100, size=n)})
    df['category'] = Series(np.array(list('abcdefghij')).take(np.random.randint(0, 10, size=n))).astype('category')
    df.isna()
    buf = StringIO()
    df.info(buf=buf)
    df2 = df[df['category'] == 'd']
    buf = StringIO()
    df2.info(buf=buf)