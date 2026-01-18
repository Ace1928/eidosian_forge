from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def cmd_STAT(self, line):
    """
        Reception of one stat line.
        """
    cmd = self._current[0]
    key, val = line.split(b' ', 1)
    cmd.values[key] = val