import datetime
import logging
import iso8601
def before_utcnow(**timedelta_kwargs):
    """Calculate the time in the past from utcnow.

    :param \\*\\*timedelta_kwargs:
        Passed directly to :class:`datetime.timedelta` to subtract from the
        current time in UTC.
    :returns:
        The time in the past based on ``timedelta_kwargs``.
    :rtype:
        datetime.datetime
    """
    now = datetime.datetime.utcnow()
    delta = datetime.timedelta(**timedelta_kwargs)
    return now - delta