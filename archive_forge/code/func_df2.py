import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.fixture
def df2():
    return DataFrame({'outer': [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3], 'inner': [1, 2, 2, 3, 3, 4, 2, 3, 1, 1, 2, 3], 'v2': np.linspace(10, 11, 12)})