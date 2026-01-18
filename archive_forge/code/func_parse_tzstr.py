from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
@_takes_ascii
def parse_tzstr(self, tzstr, zero_as_utc=True):
    """
        Parse a valid ISO time zone string.

        See :func:`isoparser.isoparse` for details on supported formats.

        :param tzstr:
            A string representing an ISO time zone offset

        :param zero_as_utc:
            Whether to return :class:`dateutil.tz.tzutc` for zero-offset zones

        :return:
            Returns :class:`dateutil.tz.tzoffset` for offsets and
            :class:`dateutil.tz.tzutc` for ``Z`` and (if ``zero_as_utc`` is
            specified) offsets equivalent to UTC.
        """
    return self._parse_tzstr(tzstr, zero_as_utc=zero_as_utc)