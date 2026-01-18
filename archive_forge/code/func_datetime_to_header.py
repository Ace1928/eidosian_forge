from __future__ import annotations
import calendar
import time
from datetime import datetime, timedelta, timezone
from email.utils import formatdate, parsedate, parsedate_tz
from typing import TYPE_CHECKING, Any, Mapping
def datetime_to_header(dt: datetime) -> str:
    return formatdate(calendar.timegm(dt.timetuple()))