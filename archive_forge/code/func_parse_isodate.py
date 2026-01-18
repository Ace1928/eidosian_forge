from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
@_takes_ascii
def parse_isodate(self, datestr):
    """
        Parse the date portion of an ISO string.

        :param datestr:
            The string portion of an ISO string, without a separator

        :return:
            Returns a :class:`datetime.date` object
        """
    components, pos = self._parse_isodate(datestr)
    if pos < len(datestr):
        raise ValueError('String contains unknown ISO ' + 'components: {!r}'.format(datestr.decode('ascii')))
    return date(*components)