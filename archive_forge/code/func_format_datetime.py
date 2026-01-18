import os
import re
import time
import random
import socket
import datetime
import urllib.parse
from email._parseaddr import quote
from email._parseaddr import AddressList as _AddressList
from email._parseaddr import mktime_tz
from email._parseaddr import parsedate, parsedate_tz, _parsedate_tz
from email.charset import Charset
def format_datetime(dt, usegmt=False):
    """Turn a datetime into a date string as specified in RFC 2822.

    If usegmt is True, dt must be an aware datetime with an offset of zero.  In
    this case 'GMT' will be rendered instead of the normal +0000 required by
    RFC2822.  This is to support HTTP headers involving date stamps.
    """
    now = dt.timetuple()
    if usegmt:
        if dt.tzinfo is None or dt.tzinfo != datetime.timezone.utc:
            raise ValueError('usegmt option requires a UTC datetime')
        zone = 'GMT'
    elif dt.tzinfo is None:
        zone = '-0000'
    else:
        zone = dt.strftime('%z')
    return _format_timetuple_and_zone(now, zone)