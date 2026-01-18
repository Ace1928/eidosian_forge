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
@onlyOnPOSIX
def test_errorDuringExec(self):
    """
        When L{os.execvpe} raises an exception, it will format that exception
        on stderr as UTF-8, regardless of system encoding information.
        """

    def execvpe(*args, **kw):
        filename = '<☃>'
        if not isinstance(filename, str):
            filename = filename.encode('utf-8')
        codeobj = compile('1/0', filename, 'single')
        eval(codeobj)
    self.patch(os, 'execvpe', execvpe)
    self.patch(sys, 'getfilesystemencoding', lambda: 'ascii')
    reactor = self.buildReactor()
    reactor._neverUseSpawn = True
    output = io.BytesIO()
    expectedFD = 1 if self.usePTY else 2

    @reactor.callWhenRunning
    def whenRunning():

        class TracebackCatcher(ProcessProtocol):

            def childDataReceived(self, child, data):
                if child == expectedFD:
                    output.write(data)

            def processEnded(self, reason):
                reactor.stop()
        reactor.spawnProcess(TracebackCatcher(), pyExe, [pyExe, b'-c', b''], usePTY=self.usePTY)
    self.runReactor(reactor, timeout=30)
    self.assertIn('☃'.encode(), output.getvalue())