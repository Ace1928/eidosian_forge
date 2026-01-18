import urllib
from oslo_log import log as logging
from oslo_utils import timeutils
from glance.common import exception
from glance.i18n import _, _LE
@property
def callback_due(self):
    """Indicates if a callback should be made.

        If no time-based limit is set, this will always be True.
        If a limit is set, then this returns True exactly once,
        resetting the timer when it does.
        """
    if not self._min_interval:
        return True
    if not self._timer:
        self._timer = timeutils.StopWatch(self._min_interval)
        self._timer.start()
    if self._timer.expired():
        self._timer.restart()
        return True
    else:
        return False