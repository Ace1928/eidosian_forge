from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
def previous_friday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt