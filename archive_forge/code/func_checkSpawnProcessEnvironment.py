import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
def checkSpawnProcessEnvironment(self, spawnKwargs, expectedEnv, usePosixSpawnp):
    """
        Shared code for testing the environment variables
        present in the spawned process.

        The spawned process serializes its environ to stderr or stdout (depending on usePTY)
        which is checked against os.environ of the calling process.
        """
    p = Accumulator()
    d = p.endedDeferred = Deferred()
    reactor = self.buildReactor()
    reactor._neverUseSpawn = not usePosixSpawnp
    reactor.callWhenRunning(reactor.spawnProcess, p, pyExe, [pyExe, b'-c', networkString('import os, sys; env = dict(os.environ); env.pop("LC_CTYPE", None); env.pop("__CF_USER_TEXT_ENCODING", None); sys.stderr.write(str(sorted(env.items())))')], usePTY=self.usePTY, **spawnKwargs)

    def shutdown(ign):
        reactor.stop()
    d.addBoth(shutdown)
    self.runReactor(reactor)
    expectedEnv.pop('LC_CTYPE', None)
    expectedEnv.pop('__CF_USER_TEXT_ENCODING', None)
    self.assertEqual(bytes(str(sorted(expectedEnv.items())), 'utf-8'), p.outF.getvalue() if self.usePTY else p.errF.getvalue())