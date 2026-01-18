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
@property
def naive(self) -> dt_datetime:
    """Returns a naive datetime representation of the :class:`Arrow <arrow.arrow.Arrow>`
        object.

        Usage::

            >>> nairobi = arrow.now('Africa/Nairobi')
            >>> nairobi
            <Arrow [2019-01-23T19:27:12.297999+03:00]>
            >>> nairobi.naive
            datetime.datetime(2019, 1, 23, 19, 27, 12, 297999)

        """
    return self._datetime.replace(tzinfo=None)