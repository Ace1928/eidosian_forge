import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
def test_deliveryRejectedSenderAddress(self):
    """
        Test that a C{MAIL FROM} command with an address rejected by a
        L{smtp.IMessageDelivery} instance is responded to with the correct
        error code.
        """

    class RejectionDelivery(NotImplementedDelivery):
        """
            Delivery object which rejects all senders as invalid.
            """

        def validateFrom(self, helo, origin):
            raise smtp.SMTPBadSender(origin)
    realm = SingletonRealm(smtp.IMessageDelivery, RejectionDelivery())
    portal = Portal(realm, [AllowAnonymousAccess()])
    proto = smtp.SMTP()
    proto.portal = portal
    trans = StringTransport()
    proto.makeConnection(trans)
    proto.dataReceived(b'HELO example.com\r\n')
    trans.clear()
    proto.dataReceived(b'MAIL FROM:<alice@example.com>\r\n')
    proto.connectionLost(error.ConnectionLost())
    self.assertEqual(trans.value(), b'550 Cannot receive from specified address <alice@example.com>: Sender not acceptable\r\n')