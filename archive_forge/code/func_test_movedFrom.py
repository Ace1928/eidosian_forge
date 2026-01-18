import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_movedFrom(self):
    """
        Moving a file out of a monitored directory sends an
        C{inotify.IN_MOVED_FROM} event to the callback.
        """

    def operation(path):
        path.open('w').close()
        path.moveTo(filepath.FilePath(self.mktemp()))
    return self._notificationTest(inotify.IN_MOVED_FROM, operation)