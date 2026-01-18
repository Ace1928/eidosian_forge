import calendar
from datetime import datetime, timedelta
from twisted.python.compat import nativeString
from twisted.python.util import FancyStrMixin
@classmethod
def fromRFC4034DateString(cls, utcDateString):
    """
        Create an L{SerialNumber} instance from a date string in format
        'YYYYMMDDHHMMSS' described in U{RFC4034
        3.2<https://tools.ietf.org/html/rfc4034#section-3.2>}.

        The L{SerialNumber} instance stores the date as a 32bit UNIX timestamp.

        @see: U{https://tools.ietf.org/html/rfc4034#section-3.1.5}

        @param utcDateString: A UTC date/time string of format I{YYMMDDhhmmss}
            which will be converted to seconds since the UNIX epoch.
        @type utcDateString: L{unicode}

        @return: An L{SerialNumber} instance containing the supplied date as a
            32bit UNIX timestamp.
        """
    parsedDate = datetime.strptime(utcDateString, RFC4034_TIME_FORMAT)
    secondsSinceEpoch = calendar.timegm(parsedDate.utctimetuple())
    return cls(secondsSinceEpoch, serialBits=32)