import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_movedTo(self):
    """
        Moving a file into a monitored directory sends an
        C{inotify.IN_MOVED_TO} event to the callback.
        """

    def operation(path):
        p = filepath.FilePath(self.mktemp())
        p.touch()
        p.moveTo(path)
    return self._notificationTest(inotify.IN_MOVED_TO, operation)