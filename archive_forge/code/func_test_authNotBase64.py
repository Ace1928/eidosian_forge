from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_authNotBase64(self):
    """
        A client that responds with a challenge that cannot be decoded
        as Base 64 receives an L{IllegalClientResponse}.
        """

    @implementer(IChallengeResponse)
    class NotBase64AuthChallenge:
        message = b'Malformed Response - not base64'

        def getChallenge(self):
            return b'SomeChallenge'

        def setResponse(self, response):
            """
                Never called.

                @param response: See L{IChallengeResponse.setResponse}
                """

        def moreChallenges(self):
            """
                Never called.
                """
    notBase64 = NotBase64AuthChallenge()
    verifyObject(IChallengeResponse, notBase64)
    server = imap4.IMAP4Server()
    server.portal = self.portal
    server.challengers[b'NOTBASE64'] = NotBase64AuthChallenge
    transport = StringTransport()
    server.makeConnection(transport)
    self.addCleanup(server.connectionLost, error.ConnectionDone('Connection done.'))
    self.assertIn(b'AUTH=NOTBASE64', transport.value())
    transport.clear()
    server.dataReceived(b'001 AUTHENTICATE NOTBASE64\r\n')
    self.assertIn(base64.b64encode(notBase64.getChallenge()), transport.value())
    transport.clear()
    server.dataReceived(b'\x00 Not base64\r\n')
    self.assertEqual(transport.value(), b''.join([b'001 NO Authentication failed: ', notBase64.message, b'\r\n']))