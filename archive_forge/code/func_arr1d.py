from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.fixture
def arr1d(self, period_index):
    """
        Fixture returning DatetimeArray from parametrized PeriodIndex objects
        """
    return period_index._data