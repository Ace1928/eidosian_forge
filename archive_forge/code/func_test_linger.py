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
def test_linger(self):
    scriptPath = b'twisted.test.process_linger'
    p = Accumulator()
    d = p.endedDeferred = defer.Deferred()
    reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv, childFDs={1: 'r', 2: 2})

    def processEnded(ign):
        self.assertEqual(p.outF.getvalue(), b'here is some text\ngoodbye\n')
    return d.addCallback(processEnded)