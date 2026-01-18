from datetime import datetime
import numpy as np
import pytest
from pandas import (
@pytest.fixture
def empty_frame_dti(series):
    """
    Fixture for parametrization of empty DataFrame with date_range,
    period_range and timedelta_range indexes
    """
    index = series.index[:0]
    return DataFrame(index=index)