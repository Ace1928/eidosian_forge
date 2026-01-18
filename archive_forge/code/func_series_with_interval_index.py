import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def series_with_interval_index(self):
    return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))