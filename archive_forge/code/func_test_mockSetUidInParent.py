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
def test_mockSetUidInParent(self):
    """
        When spawning a child process with a UID different from the UID of the
        current process, the current process does not have its UID changed.
        """
    self.mockos.child = False
    cmd = b'/mock/ouch'
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False, uid=8080)
    self.assertProcessLaunched()