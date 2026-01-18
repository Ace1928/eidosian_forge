import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def test_mockForkErrorGivenFDs(self):
    """
        When C{os.forks} raises an exception and that file descriptors have
        been specified with the C{childFDs} arguments of
        L{reactor.spawnProcess}, they are not closed.
        """
    self.mockos.raiseFork = OSError(errno.EAGAIN, None)
    protocol = TrivialProcessProtocol(None)
    self.assertRaises(OSError, reactor.spawnProcess, protocol, None, childFDs={0: -10, 1: -11, 2: -13})
    self.assertEqual(self.mockos.actions, [('fork', False)])
    self.assertEqual(self.mockos.closed, [])
    self.assertRaises(OSError, reactor.spawnProcess, protocol, None, childFDs={0: 'r', 1: -11, 2: -13})
    self.assertEqual(set(self.mockos.closed), {-1, -2})