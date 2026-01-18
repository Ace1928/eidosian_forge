from __future__ import absolute_import
import time
from socket import _GLOBAL_DEFAULT_TIMEOUT
from ..exceptions import TimeoutStateError
def start_connect(self):
    """Start the timeout clock, used during a connect() attempt

        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to start a timer that has been started already.
        """
    if self._start_connect is not None:
        raise TimeoutStateError('Timeout timer has already been started.')
    self._start_connect = current_time()
    return self._start_connect