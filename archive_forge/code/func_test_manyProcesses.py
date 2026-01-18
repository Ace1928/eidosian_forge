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
@skipIf(os.environ.get('CI', '').lower() == 'true' and runtime.platform.getType() == 'win32', 'See https://twistedmatrix.com/trac/ticket/10014')
def test_manyProcesses(self):

    def _check(results, protocols):
        for p in protocols:
            self.assertEqual(p.stages, [1, 2, 3, 4, 5], '[%d] stages = %s' % (id(p.transport), str(p.stages)))
            f = p.reason
            f.trap(error.ProcessTerminated)
            self.assertEqual(f.value.exitCode, 23)
    scriptPath = b'twisted.test.process_tester'
    args = [pyExe, b'-u', b'-m', scriptPath]
    protocols = []
    deferreds = []
    for i in range(CONCURRENT_PROCESS_TEST_COUNT):
        p = TestManyProcessProtocol()
        protocols.append(p)
        reactor.spawnProcess(p, pyExe, args, env=properEnv)
        deferreds.append(p.deferred)
    deferredList = defer.DeferredList(deferreds, consumeErrors=True)
    deferredList.addCallback(_check, protocols)
    return deferredList