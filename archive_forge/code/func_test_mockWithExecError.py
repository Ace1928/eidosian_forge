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
def test_mockWithExecError(self):
    """
        Spawn a process but simulate an error during execution in the client
        path: C{os.execvpe} raises an error. It should close all the standard
        fds, try to print the error encountered, and exit cleanly.
        """
    cmd = b'/mock/ouch'
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    self.mockos.raiseExec = True
    try:
        reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
    except SystemError:
        self.assertTrue(self.mockos.exited)
        self.assertEqual(self.mockos.actions, [('fork', False), 'exec', ('exit', 1)])
        self.assertIn(0, self.mockos.closed)
        self.assertIn(1, self.mockos.closed)
        self.assertIn(2, self.mockos.closed)
        self.assertIn(b'RuntimeError: Bar', self.mockos.fdio.getvalue())
    else:
        self.fail('Should not be here')