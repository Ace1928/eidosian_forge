from __future__ import annotations
import datetime
from lazyops.imports._dateparser import (
from typing import Optional, List, Union
def convert_possible_date(dt: Optional[str]=None) -> Optional[datetime.datetime]:
    """
    Converts a possible date string into a datetime
    """
    if not dt:
        return None
    dt = dt.strip()
    if not dt:
        return None
    resolve_dateparser(True)
    if dt[:2].isalpha():
        _tz = None
        if dt[-2:] not in {'AM', 'PM'}:
            dt, _tz = dt.rsplit(' ', 1)
            _tz = tz_map.get(_tz)
            if _tz:
                _tz = pytz.timezone(_tz)
        dt_val = datetime.datetime.strptime(dt, '%b %d, %Y %H:%M:%S %p')
        if _tz:
            dt_val = _tz.localize(dt_val).astimezone(pytz.utc)
        return dt_val
    if 'T' in dt:
        if '+' in dt:
            return datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S%z')
        if '-' in dt[-7:]:
            dt = dt.rsplit('-', 1)[0]
        return datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S')
    if ' ' in dt and '.' in dt:
        dt = dt.split('.')[0]
        return datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    if '-' in dt:
        return datetime.datetime.strptime(dt, '%Y-%m-%d')
    if '/' in dt:
        return datetime.datetime.strptime(dt, '%m/%d/%Y')
    return datetime.datetime.strptime(dt, '%m%d%Y')