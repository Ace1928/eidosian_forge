import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_deleteSelf(self):
    """
        Deleting the monitored directory itself sends an
        C{inotify.IN_DELETE_SELF} event to the callback.
        """

    def operation(path):
        path.remove()
    return self._notificationTest(inotify.IN_DELETE_SELF, operation, expectedPath=self.dirname)