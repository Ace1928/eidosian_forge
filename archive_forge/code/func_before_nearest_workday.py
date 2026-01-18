from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
def before_nearest_workday(dt: datetime) -> datetime:
    """
    returns previous workday after nearest workday
    """
    return previous_workday(nearest_workday(dt))