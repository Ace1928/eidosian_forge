import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
def test_nonZeroExitSignal(self):
    """
        When the command exits with a non-zero signal, the protocol's
        C{connectionLost} method is called with a L{Failure} wrapping an
        exception which encapsulates that status.

        Additional packet contents are logged at the C{info} level.
        """
    logObserver = EventLoggingObserver()
    globalLogPublisher.addObserver(logObserver)
    self.addCleanup(globalLogPublisher.removeObserver, logObserver)
    exitCode = None
    signal = 15
    packet = b''.join([common.NS(b'TERM'), b'\x01', common.NS(b'message'), common.NS(b'en-US')])
    exc = self._exitStatusTest(b'exit-signal', packet)
    exc.trap(ProcessTerminated)
    self.assertEqual(exitCode, exc.value.exitCode)
    self.assertEqual(signal, exc.value.signal)
    logNamespace = 'twisted.conch.endpoints._CommandChannel'
    hamcrest.assert_that(logObserver, hamcrest.has_item(hamcrest.has_entries({'log_level': hamcrest.equal_to(LogLevel.info), 'log_namespace': logNamespace, 'shortSignalName': b'TERM', 'coreDumped': True, 'errorMessage': 'message', 'languageTag': b'en-US'})))