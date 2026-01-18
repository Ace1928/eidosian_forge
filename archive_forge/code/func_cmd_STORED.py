from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_STORED(self):
    """
        Manage a success response to a set operation.
        """
    self._current.popleft().success(True)