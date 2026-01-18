import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_initializationErrors(self):
    """
        L{inotify.INotify} emits a C{RuntimeError} when initialized
        in an environment that doesn't support inotify as we expect it.

        We just try to raise an exception for every possible case in
        the for loop in L{inotify.INotify._inotify__init__}.
        """

    class FakeINotify:

        def init(self):
            raise inotify.INotifyError()
    self.patch(inotify.INotify, '_inotify', FakeINotify())
    self.assertRaises(inotify.INotifyError, inotify.INotify)