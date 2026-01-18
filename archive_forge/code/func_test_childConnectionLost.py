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
def test_childConnectionLost(self):
    """
        L{IProcessProtocol.childConnectionLost} is called each time a file
        descriptor associated with a child process is closed.
        """
    connected = Deferred()
    lost = {0: Deferred(), 1: Deferred(), 2: Deferred()}

    class Closer(ProcessProtocol):

        def makeConnection(self, transport):
            connected.callback(transport)

        def childConnectionLost(self, childFD):
            lost[childFD].callback(None)
    target = b'twisted.internet.test.process_loseconnection'
    reactor = self.buildReactor()
    reactor.callWhenRunning(reactor.spawnProcess, Closer(), pyExe, [pyExe, b'-m', target], env=properEnv, usePTY=self.usePTY)

    def cbConnected(transport):
        transport.write(b'2\n')
        return lost[2].addCallback(lambda ign: transport)
    connected.addCallback(cbConnected)

    def lostSecond(transport):
        transport.write(b'1\n')
        return lost[1].addCallback(lambda ign: transport)
    connected.addCallback(lostSecond)

    def lostFirst(transport):
        transport.write(b'\n')
    connected.addCallback(lostFirst)
    connected.addErrback(err)

    def cbEnded(ignored):
        reactor.stop()
    connected.addCallback(cbEnded)
    self.runReactor(reactor)