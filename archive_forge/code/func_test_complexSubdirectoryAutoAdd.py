import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_complexSubdirectoryAutoAdd(self):
    """
        L{inotify.INotify} with autoAdd==True for a watched path
        generates events for every file or directory already present
        in a newly created subdirectory under the watched one.

        This tests that we solve a race condition in inotify even though
        we may generate duplicate events.
        """
    calls = set()

    def _callback(wp, filename, mask):
        calls.add(filename)
        if len(calls) == 6:
            try:
                self.assertTrue(self.inotify._isWatched(subdir))
                self.assertTrue(self.inotify._isWatched(subdir2))
                self.assertTrue(self.inotify._isWatched(subdir3))
                created = someFiles + [subdir, subdir2, subdir3]
                created = {f.asBytesMode() for f in created}
                self.assertEqual(len(calls), len(created))
                self.assertEqual(calls, created)
            except Exception:
                d.errback()
            else:
                d.callback(None)
    checkMask = inotify.IN_ISDIR | inotify.IN_CREATE
    self.inotify.watch(self.dirname, mask=checkMask, autoAdd=True, callbacks=[_callback])
    subdir = self.dirname.child('test')
    subdir2 = subdir.child('test2')
    subdir3 = subdir2.child('test3')
    d = defer.Deferred()
    subdir3.makedirs()
    someFiles = [subdir.child('file1.dat'), subdir2.child('file2.dat'), subdir3.child('file3.dat')]
    for i, filename in enumerate(someFiles):
        filename.setContent(filename.path.encode(sys.getfilesystemencoding()))
    return d