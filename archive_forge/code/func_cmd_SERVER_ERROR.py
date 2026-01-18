from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_SERVER_ERROR(self, errText):
    """
        An error has happened server-side.
        """
    errText = repr(errText)
    log.err('Server error: ' + errText)
    cmd = self._current.popleft()
    cmd.fail(ServerError(errText))