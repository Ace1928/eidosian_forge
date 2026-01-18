import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_noAutoAddSubdirectory(self):
    """
        L{inotify.INotify.watch} with autoAdd==False will stop inotify
        from watching subdirectories created under the watched one.
        """

    def _callback(wp, fp, mask):

        def _():
            try:
                self.assertFalse(self.inotify._isWatched(subdir))
                d.callback(None)
            except Exception:
                d.errback()
        reactor.callLater(0, _)
    checkMask = inotify.IN_ISDIR | inotify.IN_CREATE
    self.inotify.watch(self.dirname, mask=checkMask, autoAdd=False, callbacks=[_callback])
    subdir = self.dirname.child('test')
    d = defer.Deferred()
    subdir.createDirectory()
    return d