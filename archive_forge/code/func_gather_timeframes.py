import calendar
import re
import sys
from datetime import date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta
from datetime import tzinfo as dt_tzinfo
from math import trunc
from time import struct_time
from typing import (
from dateutil import tz as dateutil_tz
from dateutil.relativedelta import relativedelta
from arrow import formatter, locales, parser, util
from arrow.constants import DEFAULT_LOCALE, DEHUMANIZE_LOCALES
from arrow.locales import TimeFrameLiteral
def gather_timeframes(_delta: float, _frame: TimeFrameLiteral) -> float:
    if _frame in granularity:
        value = sign * _delta / self._SECS_MAP[_frame]
        _delta %= self._SECS_MAP[_frame]
        if trunc(abs(value)) != 1:
            timeframes.append((cast(TimeFrameLiteral, _frame + 's'), value))
        else:
            timeframes.append((_frame, value))
    return _delta