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
def test_commandLine(self):
    args = [b'a\\"b ', b'a\\b ', b' a\\\\"b', b' a\\\\b', b'"foo bar" "', b'\tab', b'"\\', b'a"b', b"a'b"]
    scriptPath = b'twisted.test.process_cmdline'
    p = Accumulator()
    d = p.endedDeferred = defer.Deferred()
    reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath] + args, env=properEnv, path=None)

    def processEnded(ign):
        self.assertEqual(p.errF.getvalue(), b'')
        recvdArgs = p.outF.getvalue().splitlines()
        self.assertEqual(recvdArgs, args)
    return d.addCallback(processEnded)