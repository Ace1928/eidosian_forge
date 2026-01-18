import threading
from twisted.python import log
from ._util import synchronized
@synchronized
def log_errors(self):
    """
        Log errors for all stored EventualResults that have error results.
        """
    for result in self._stored.values():
        failure = result.original_failure()
        if failure is not None:
            log.err(failure, 'Unhandled error in stashed EventualResult:')