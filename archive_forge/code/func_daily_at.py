import datetime
import numbers
import abc
import bisect
import pytz
@classmethod
def daily_at(cls, at, target):
    """
        Schedule a command to run at a specific time each day.

        >>> from tempora import utc
        >>> noon = utc.time(12, 0)
        >>> cmd = PeriodicCommandFixedDelay.daily_at(noon, None)
        >>> cmd.delay.total_seconds()
        86400.0
        """
    daily = datetime.timedelta(days=1)
    when = datetime.datetime.combine(datetime.date.today(), at)
    when -= daily
    while when < now():
        when += daily
    return cls.at_time(cls._localize(when), daily, target)