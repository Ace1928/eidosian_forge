from __future__ import absolute_import
import time
from socket import _GLOBAL_DEFAULT_TIMEOUT
from ..exceptions import TimeoutStateError
@property
def read_timeout(self):
    """Get the value for the read timeout.

        This assumes some time has elapsed in the connection timeout and
        computes the read timeout appropriately.

        If self.total is set, the read timeout is dependent on the amount of
        time taken by the connect timeout. If the connection time has not been
        established, a :exc:`~urllib3.exceptions.TimeoutStateError` will be
        raised.

        :return: Value to use for the read timeout.
        :rtype: int, float, :attr:`Timeout.DEFAULT_TIMEOUT` or None
        :raises urllib3.exceptions.TimeoutStateError: If :meth:`start_connect`
            has not yet been called on this object.
        """
    if self.total is not None and self.total is not self.DEFAULT_TIMEOUT and (self._read is not None) and (self._read is not self.DEFAULT_TIMEOUT):
        if self._start_connect is None:
            return self._read
        return max(0, min(self.total - self.get_connect_duration(), self._read))
    elif self.total is not None and self.total is not self.DEFAULT_TIMEOUT:
        return max(0, self.total - self.get_connect_duration())
    else:
        return self._read