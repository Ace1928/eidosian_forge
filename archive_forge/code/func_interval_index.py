import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture
def interval_index():
    return IntervalIndex.from_tuples([(0.0, 1.0), (1.0, 2.0), (1.5, 2.5)], name='interval_index')