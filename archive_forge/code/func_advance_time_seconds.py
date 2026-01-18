import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def advance_time_seconds(seconds):
    """Advance overridden time by seconds.

    See :py:class:`oslo_utils.fixture.TimeFixture`.

    """
    advance_time_delta(datetime.timedelta(0, seconds))