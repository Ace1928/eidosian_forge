import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
def getOffsetFromUtc():
    """Retrieve the utc offset respecting the daylight saving time"""
    ts = time.localtime()
    if ts.tm_isdst:
        utc_offset = time.altzone
    else:
        utc_offset = time.timezone
    return utc_offset