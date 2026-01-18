import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.fixture
def roll_frame():
    return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40)})