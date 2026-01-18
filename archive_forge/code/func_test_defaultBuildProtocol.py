from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_defaultBuildProtocol(self):
    """
        L{Factory.buildProtocol} by default constructs a protocol by calling
        its C{protocol} attribute, and attaches the factory to the result.
        """

    class SomeProtocol(Protocol):
        pass
    f = Factory()
    f.protocol = SomeProtocol
    protocol = f.buildProtocol(None)
    self.assertIsInstance(protocol, SomeProtocol)
    self.assertIs(protocol.factory, f)