import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_messageReceivedError(self):
    """
        When a client with SSL enabled talks to a server without SSL, it
        should return a meaningful error.
        """
    svr = SecurableProto()
    okc = OKCert()
    svr.certFactory = lambda: okc
    box = amp.Box()
    box[b'_command'] = b'StartTLS'
    box[b'_ask'] = b'1'
    boxes = []
    svr.sendBox = boxes.append
    svr.makeConnection(StringTransport())
    svr.ampBoxReceived(box)
    self.assertEqual(boxes, [{b'_error_code': b'TLS_ERROR', b'_error': b'1', b'_error_description': b'TLS not available'}])