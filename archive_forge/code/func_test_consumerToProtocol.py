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
def test_consumerToProtocol(self):
    """
        L{IConsumer} providers can be adapted to L{IProtocol} providers using
        L{ProtocolToConsumerAdapter}.
        """
    result = []

    @implementer(IConsumer)
    class Consumer:

        def write(self, d):
            result.append(d)
    c = Consumer()
    protocol = IProtocol(c)
    protocol.dataReceived(b'hello')
    self.assertEqual(result, [b'hello'])
    self.assertIsInstance(protocol, ConsumerToProtocolAdapter)