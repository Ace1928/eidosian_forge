from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
def open_and_lock(self, timeout=0, delay=0.05):
    """Open the file, trying to lock it.

        Args:
            timeout: float, The number of seconds to try to acquire the lock.
            delay: float, The number of seconds to wait between retry attempts.

        Raises:
            AlreadyLockedException: if the lock is already acquired.
            IOError: if the open fails.
        """
    self._opener.open_and_lock(timeout, delay)