from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
def from_timestamp(timestamp: float, tz_offset: float) -> datetime:
    """Convert a timestamp + tz_offset into an aware datetime instance."""
    utc_dt = datetime.fromtimestamp(timestamp, utc)
    try:
        local_dt = utc_dt.astimezone(tzoffset(tz_offset))
        return local_dt
    except ValueError:
        return utc_dt