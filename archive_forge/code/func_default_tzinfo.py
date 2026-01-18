from __future__ import unicode_literals
from datetime import datetime, time
def default_tzinfo(dt, tzinfo):
    """
    Sets the ``tzinfo`` parameter on naive datetimes only

    This is useful for example when you are provided a datetime that may have
    either an implicit or explicit time zone, such as when parsing a time zone
    string.

    .. doctest::

        >>> from dateutil.tz import tzoffset
        >>> from dateutil.parser import parse
        >>> from dateutil.utils import default_tzinfo
        >>> dflt_tz = tzoffset("EST", -18000)
        >>> print(default_tzinfo(parse('2014-01-01 12:30 UTC'), dflt_tz))
        2014-01-01 12:30:00+00:00
        >>> print(default_tzinfo(parse('2014-01-01 12:30'), dflt_tz))
        2014-01-01 12:30:00-05:00

    :param dt:
        The datetime on which to replace the time zone

    :param tzinfo:
        The :py:class:`datetime.tzinfo` subclass instance to assign to
        ``dt`` if (and only if) it is naive.

    :return:
        Returns an aware :py:class:`datetime.datetime`.
    """
    if dt.tzinfo is not None:
        return dt
    else:
        return dt.replace(tzinfo=tzinfo)