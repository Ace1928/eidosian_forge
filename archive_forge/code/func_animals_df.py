from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture
def animals_df():
    return DataFrame({'key': [1, 1, 1, 1], 'num_legs': [2, 4, 4, 6], 'num_wings': [2, 0, 0, 0]}, index=['falcon', 'dog', 'cat', 'ant'])