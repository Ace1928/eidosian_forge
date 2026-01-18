import collections
import logging
import threading
import time
import types
def set_running_or_notify_cancel(self):
    """Mark the future as running or process any cancel notifications.

        Should only be used by Executor implementations and unit tests.

        If the future has been cancelled (cancel() was called and returned
        True) then any threads waiting on the future completing (though calls
        to as_completed() or wait()) are notified and False is returned.

        If the future was not cancelled then it is put in the running state
        (future calls to running() will return True) and True is returned.

        This method should be called by Executor implementations before
        executing the work associated with this future. If this method returns
        False then the work should not be executed.

        Returns:
            False if the Future was cancelled, True otherwise.

        Raises:
            RuntimeError: if this method was already called or if set_result()
                or set_exception() was called.
        """
    with self._condition:
        if self._state == CANCELLED:
            self._state = CANCELLED_AND_NOTIFIED
            for waiter in self._waiters:
                waiter.add_cancelled(self)
            return False
        elif self._state == PENDING:
            self._state = RUNNING
            return True
        else:
            LOGGER.critical('Future %s in unexpected state: %s', id(self), self._state)
            raise RuntimeError('Future in unexpected state')