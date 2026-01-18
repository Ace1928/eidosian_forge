import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_ignoreDirectory(self):
    """
        L{inotify.INotify.ignore} removes a directory from the watchlist
        """
    self.inotify.watch(self.dirname, autoAdd=True)
    self.assertTrue(self.inotify._isWatched(self.dirname))
    self.inotify.ignore(self.dirname)
    self.assertFalse(self.inotify._isWatched(self.dirname))