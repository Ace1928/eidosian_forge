from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_NOT_STORED(self):
    """
        Manage a specific 'not stored' response to a set operation: this is not
        an error, but some condition wasn't met.
        """
    self._current.popleft().success(False)