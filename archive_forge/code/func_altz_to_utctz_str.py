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
def altz_to_utctz_str(altz: float) -> str:
    """Convert a timezone offset west of UTC in seconds into a Git timezone offset string.

    :param altz: Timezone offset in seconds west of UTC
    """
    hours = abs(altz) // 3600
    minutes = abs(altz) % 3600 // 60
    sign = '-' if altz >= 60 else '+'
    return '{}{:02}{:02}'.format(sign, hours, minutes)