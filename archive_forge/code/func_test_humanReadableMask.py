import sys
from twisted.internet import defer, reactor
from twisted.python import filepath, runtime
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_humanReadableMask(self):
    """
        L{inotify.humaReadableMask} translates all the possible event
        masks to a human readable string.
        """
    for mask, value in inotify._FLAG_TO_HUMAN:
        self.assertEqual(inotify.humanReadableMask(mask)[0], value)
    checkMask = inotify.IN_CLOSE_WRITE | inotify.IN_ACCESS | inotify.IN_OPEN
    self.assertEqual(set(inotify.humanReadableMask(checkMask)), {'close_write', 'access', 'open'})