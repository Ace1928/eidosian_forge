import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.fixture
def df_with_string_col():
    df = DataFrame({'a': [1, 1, 1, 1, 1, 2, 2, 2, 2], 'b': [3, 3, 4, 4, 4, 4, 4, 3, 3], 'c': range(9), 'd': list('xyzwtyuio')})
    return df