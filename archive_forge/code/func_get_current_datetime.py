from __future__ import annotations
import datetime
from lazyops.imports._dateparser import (
from typing import Optional, List, Union
def get_current_datetime(tz: Optional[str]=None) -> datetime.datetime:
    """
    Gets the current datetime in the specified timezone
    """
    resolve_dateparser(True)
    if tz:
        tz = pytz.timezone(tz_map.get(tz.upper(), tz))
    return datetime.datetime.now(tz)