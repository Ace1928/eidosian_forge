import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def set_time_override(override_time=None):
    """Overrides utils.utcnow.

    Make it return a constant time or a list thereof, one at a time.

    See :py:class:`oslo_utils.fixture.TimeFixture`.

    :param override_time: datetime instance or list thereof. If not
                          given, defaults to the current UTC time.
    """
    utcnow.override_time = override_time or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)