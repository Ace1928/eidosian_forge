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
def test_mockErrorECHILDInReapProcess(self):
    """
        Test that reapProcess doesn't log anything when waitpid raises a
        C{OSError} with errno C{ECHILD}.
        """
    self.mockos.child = False
    cmd = b'/mock/ouch'
    self.mockos.waitChild = (0, 0)
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
    self.assertProcessLaunched()
    self.mockos.raiseWaitPid = OSError()
    self.mockos.raiseWaitPid.errno = errno.ECHILD
    proc.reapProcess()