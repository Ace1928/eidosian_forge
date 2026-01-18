import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def months2days(year: int, month: int, months_delta: int) -> int:
    """
    Converts a delta of months to a delta of days, counting from the 1st day of the month,
    relative to the year and the month passed as arguments.

    :param year: the reference start year, a negative or zero value means a BCE year     (0 is 1 BCE, -1 is 2 BCE, -2 is 3 BCE, etc.).
    :param month: the starting month (1-12).
    :param months_delta: the number of months, if negative count backwards.
    """
    if not months_delta:
        return 0
    total_months = month - 1 + months_delta
    target_year = year + total_months // 12
    target_month = total_months % 12 + 1
    if month <= 2:
        y_days = 365 * (target_year - year) + leapdays(year, target_year)
    else:
        y_days = 365 * (target_year - year) + leapdays(year + 1, target_year + 1)
    months_days = MONTH_DAYS_LEAP if isleap(target_year) else MONTH_DAYS
    if target_month >= month:
        m_days = sum((months_days[m] for m in range(month, target_month)))
        return y_days + m_days if y_days >= 0 else y_days + m_days
    else:
        m_days = sum((months_days[m] for m in range(target_month, month)))
        return y_days - m_days if y_days >= 0 else y_days - m_days