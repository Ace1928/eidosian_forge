from __future__ import absolute_import
import time
from socket import _GLOBAL_DEFAULT_TIMEOUT
from ..exceptions import TimeoutStateError
@property
def connect_timeout(self):
    """Get the value to use when setting a connection timeout.

        This will be a positive float or integer, the value None
        (never timeout), or the default system timeout.

        :return: Connect timeout.
        :rtype: int, float, :attr:`Timeout.DEFAULT_TIMEOUT` or None
        """
    if self.total is None:
        return self._connect
    if self._connect is None or self._connect is self.DEFAULT_TIMEOUT:
        return self.total
    return min(self._connect, self.total)