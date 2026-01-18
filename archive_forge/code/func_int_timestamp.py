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
def int_timestamp(self) -> int:
    """Returns an integer timestamp representation of the :class:`Arrow <arrow.arrow.Arrow>` object, in
        UTC time.

        Usage::

            >>> arrow.utcnow().int_timestamp
            1548260567

        """
    return int(self.timestamp())