from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture
def df_duplicates():
    return pd.DataFrame({'a': [1, 2, 3, 4, 4], 'b': [1, 1, 1, 1, 1], 'c': [0, 1, 2, 5, 4]}, index=[0, 0, 1, 1, 1])