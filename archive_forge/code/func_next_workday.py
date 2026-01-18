from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
def next_workday(dt: datetime) -> datetime:
    """
    returns next weekday used for observances
    """
    dt += timedelta(days=1)
    while dt.weekday() > 4:
        dt += timedelta(days=1)
    return dt