from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_ERROR(self):
    """
        A non-existent command has been sent.
        """
    log.err('Non-existent command sent.')
    cmd = self._current.popleft()
    cmd.fail(NoSuchCommand())