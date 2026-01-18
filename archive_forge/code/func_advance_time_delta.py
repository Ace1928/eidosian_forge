import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def advance_time_delta(timedelta):
    """Advance overridden time using a datetime.timedelta.

    See :py:class:`oslo_utils.fixture.TimeFixture`.

    """
    assert utcnow.override_time is not None
    try:
        for dt in utcnow.override_time:
            dt += timedelta
    except TypeError:
        utcnow.override_time += timedelta