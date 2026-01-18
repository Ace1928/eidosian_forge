import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.fixture
def index_large():
    large = [2 ** 63, 2 ** 63 + 10, 2 ** 63 + 15, 2 ** 63 + 20, 2 ** 63 + 25]
    return Index(large, dtype=np.uint64)