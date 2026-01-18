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
def test_cloexecStayPut(self) -> None:
    """
        If a file descriptor is close-on-exec and it's left in the same place,
        then we need to DUP2 it elsewhere, close the original, then DUP2 it
        back so it doesn't get closed by the implicit exec at the end of
        posix_spawn's file actions.
        """
    self.assertEqual(_getFileActions([(0, True)], {0: 0}, CLOSE, DUP2), [(DUP2, 0, 1), (DUP2, 1, 0), (CLOSE, 1)])