from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def test_fieldContentWhitespace(self):
    """
        Leading and trailing linear whitespace is stripped from the header
        value passed to the C{headerReceived} callback.
        """
    header, protocol = self._headerTestSetup()
    value = self.sep.join([b' \t ', b' bar \t', b' \t', b''])
    protocol.dataReceived(b'X-Bar:' + value)
    protocol.dataReceived(b'X-Foo:' + value)
    protocol.dataReceived(self.sep)
    self.assertEqual(header, {b'X-Foo': b'bar', b'X-Bar': b'bar'})