from __future__ import absolute_import
import time
from socket import _GLOBAL_DEFAULT_TIMEOUT
from ..exceptions import TimeoutStateError
def get_connect_duration(self):
    """Gets the time elapsed since the call to :meth:`start_connect`.

        :return: Elapsed time in seconds.
        :rtype: float
        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to get duration for a timer that hasn't been started.
        """
    if self._start_connect is None:
        raise TimeoutStateError("Can't get connect duration for timer that has not started.")
    return current_time() - self._start_connect