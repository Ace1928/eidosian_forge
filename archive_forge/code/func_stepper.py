import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
def stepper(val, n, first: bool):
    if val < MIN_REGULAR_TIMESTAMP or val > MAX_REGULAR_TIMESTAMP:
        return np.inf
    d = utcfromtimestamp(val)
    next_year = (d.year // (n * stepSize) + 1) * (n * stepSize)
    if next_year > 9999:
        return np.inf
    next_date = datetime(next_year, 1, 1)
    return (next_date - datetime(1970, 1, 1)).total_seconds()