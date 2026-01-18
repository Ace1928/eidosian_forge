import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_ignoreFilePath(self):
    """
        L{inotify.INotify} will ignore a filepath after it has been removed from
        the watch list.
        """
    expectedPath = self.dirname.child('foo.bar2')
    expectedPath.touch()
    expectedPath2 = self.dirname.child('foo.bar3')
    expectedPath2.touch()
    notified = defer.Deferred()

    def cbNotified(result):
        ignored, filename, events = result
        self.assertEqual(filename.asBytesMode(), expectedPath2.asBytesMode())
        self.assertTrue(events & inotify.IN_DELETE_SELF)

    def callIt(*args):
        notified.callback(args)
    self.assertTrue(self.inotify.watch(expectedPath, inotify.IN_DELETE_SELF, callbacks=[callIt]))
    notified.addCallback(cbNotified)
    self.assertTrue(self.inotify.watch(expectedPath2, inotify.IN_DELETE_SELF, callbacks=[callIt]))
    self.inotify.ignore(expectedPath)
    expectedPath.remove()
    expectedPath2.remove()
    return notified