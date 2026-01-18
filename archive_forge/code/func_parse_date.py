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
def parse_date(string_date: Union[str, datetime]) -> Tuple[int, int]:
    """
    Parse the given date as one of the following:

        * Aware datetime instance
        * Git internal format: timestamp offset
        * RFC 2822: Thu, 07 Apr 2005 22:13:13 +0200.
        * ISO 8601 2005-04-07T22:13:13
            The T can be a space as well.

    :return: Tuple(int(timestamp_UTC), int(offset)), both in seconds since epoch

    :raise ValueError: If the format could not be understood

    :note: Date can also be YYYY.MM.DD, MM/DD/YYYY and DD.MM.YYYY.
    """
    if isinstance(string_date, datetime):
        if string_date.tzinfo:
            utcoffset = cast(timedelta, string_date.utcoffset())
            offset = -int(utcoffset.total_seconds())
            return (int(string_date.astimezone(utc).timestamp()), offset)
        else:
            raise ValueError(f'string_date datetime object without tzinfo, {string_date}')
    try:
        if string_date.count(' ') == 1 and string_date.rfind(':') == -1:
            timestamp, offset_str = string_date.split()
            if timestamp.startswith('@'):
                timestamp = timestamp[1:]
            timestamp_int = int(timestamp)
            return (timestamp_int, utctz_to_altz(verify_utctz(offset_str)))
        else:
            offset_str = '+0000'
            if string_date[-5] in '-+':
                offset_str = verify_utctz(string_date[-5:])
                string_date = string_date[:-6]
            offset = utctz_to_altz(offset_str)
            date_formats = []
            splitter = -1
            if ',' in string_date:
                date_formats.append('%a, %d %b %Y')
                splitter = string_date.rfind(' ')
            else:
                date_formats.append('%Y-%m-%d')
                date_formats.append('%Y.%m.%d')
                date_formats.append('%m/%d/%Y')
                date_formats.append('%d.%m.%Y')
                splitter = string_date.rfind('T')
                if splitter == -1:
                    splitter = string_date.rfind(' ')
            assert splitter > -1
            time_part = string_date[splitter + 1:]
            date_part = string_date[:splitter]
            tstruct = time.strptime(time_part, '%H:%M:%S')
            for fmt in date_formats:
                try:
                    dtstruct = time.strptime(date_part, fmt)
                    utctime = calendar.timegm((dtstruct.tm_year, dtstruct.tm_mon, dtstruct.tm_mday, tstruct.tm_hour, tstruct.tm_min, tstruct.tm_sec, dtstruct.tm_wday, dtstruct.tm_yday, tstruct.tm_isdst))
                    return (int(utctime), offset)
                except ValueError:
                    continue
            raise ValueError('no format matched')
    except Exception as e:
        raise ValueError(f'Unsupported date format or type: {string_date}, type={type(string_date)}') from e