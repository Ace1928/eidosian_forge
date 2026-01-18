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
def test_mockSetUid(self):
    """
        Try creating a process with setting its uid: it's almost the same path
        as the standard path, but with a C{switchUID} call before the exec.
        """
    cmd = b'/mock/ouch'
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    try:
        reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False, uid=8080)
    except SystemError:
        self.assertTrue(self.mockos.exited)
        self.assertEqual(self.mockos.actions, [('fork', False), ('setuid', 0), ('setgid', 0), ('switchuid', 8080, 1234), 'exec', ('exit', 1)])
    else:
        self.fail('Should not be here')