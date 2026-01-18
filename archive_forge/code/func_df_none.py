import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture
def df_none():
    return DataFrame({'outer': ['a', 'a', 'a', 'b', 'b', 'b'], 'inner': [1, 2, 2, 2, 1, 1], 'A': np.arange(6, 0, -1), ('B', 5): ['one', 'one', 'two', 'two', 'one', 'one']})