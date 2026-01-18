import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_moveSelf(self):
    """
        Renaming the monitored directory itself sends an
        C{inotify.IN_MOVE_SELF} event to the callback.
        """

    def operation(path):
        path.moveTo(filepath.FilePath(self.mktemp()))
    return self._notificationTest(inotify.IN_MOVE_SELF, operation, expectedPath=self.dirname)