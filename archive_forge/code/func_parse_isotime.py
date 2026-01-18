from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
@_takes_ascii
def parse_isotime(self, timestr):
    """
        Parse the time portion of an ISO string.

        :param timestr:
            The time portion of an ISO string, without a separator

        :return:
            Returns a :class:`datetime.time` object
        """
    components = self._parse_isotime(timestr)
    if components[0] == 24:
        components[0] = 0
    return time(*components)