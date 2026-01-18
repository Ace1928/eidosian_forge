import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def adjust_day(year: int, month: int, day: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return day
    elif month in (4, 6, 9, 11):
        return min(day, 30)
    else:
        return min(day, 29) if isleap(year) else min(day, 28)